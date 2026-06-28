"""The domain training state: model, optimizer, GradScaler, epoch, plus the
config and code SHA the run was initialized from.

`TrainingState` bundles the pieces the training loop mutates and owns
serializing them — together with its config and code SHA — into a checkpoint
(`save` → `serialize_checkpoint`). The runner advances `epoch` in place and
saves at each epoch boundary; `from_checkpoint` restores the runtime state for a
resume.

A legacy-checkpoint resume also attaches a transient `WarmupSchedule` (the
`warmup` field) — it drives the optimizer, so it lives with the optimizer, but
it is deliberately not part of the checkpoint.
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
    # fp16 is the only AMP mode that needs loss scaling; the scaler is a
    # near-no-op when disabled, so the fp32 path stays unchanged.
    amp_enabled = resolve_precision(config.precision, device) == "fp16"
    return torch.amp.GradScaler(device.type, enabled=amp_enabled)


@dataclass
class TrainingState:
    """The model/optimizer/scaler/epoch the loop mutates, plus the `config` and
    `code_sha` the run was initialized from.

    `epoch` is the last *successfully completed* epoch — resume continues at
    `epoch + 1`, and a mid-epoch crash leaves it one behind (which is what
    resume expects). `config` and `code_sha` are immutable initialization facts,
    recorded into each checkpoint so it self-describes.

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
    # ramp the LR while AdamW's variance estimate re-warms. NOT written to the
    # checkpoint (`to_dict` hand-picks the runtime keys) — the resume path
    # rebuilds it per-process. None on fresh runs and combined-format resumes.
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

        Re-reads the file once for the optim/scaler/epoch payload after
        `load_checkpoint` reads it for the weights — a one-time cost paid
        only at resume startup.
        """
        # Arch comes from the checkpoint (you can't reshape restored weights);
        # `config.arch.value_head_variant` is only the legacy fallback, used
        # when the checkpoint has no `arch` key.
        model = load_checkpoint(path, device, config.arch.value_head_variant).model
        optim = _build_optim(model, config)
        scaler = _build_scaler(config, device)
        epoch = fallback_epoch

        obj = torch.load(path, map_location=device, weights_only=True)
        if is_combined_checkpoint(obj):
            if "arch" in obj:
                # Redundant second guard — the real gate is run_dir.check_drift
                # (arch is checkpoint-owned). An arch-bearing checkpoint's
                # recorded arch must match the resume config's arch.
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
        """The runtime-state portion of a checkpoint: the model/optim/scaler
        state_dicts plus `epoch`. `save` wraps this with the config and
        provenance via `serialize_checkpoint`.
        """
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
        }

    def save(self, ckpt_dir: Path) -> Path:
        """Write the checkpoint for the current epoch; return its path.

        One `torch.save` — the load either returns the whole dict or raises;
        partial writes aren't handled (operator cleans up).
        """
        path = ckpt_dir / ckpt_name(self.epoch)
        torch.save(serialize_checkpoint(self.to_dict(), self.config, self.code_sha), path)
        return path
