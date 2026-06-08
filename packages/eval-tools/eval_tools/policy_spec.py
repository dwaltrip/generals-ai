"""Parse --policy CLI arguments into Policy instances.

Formats:
    checkpoint:path/to/model.pt
    checkpoint:path/to/model.pt:force_move=true,sample=true,temperature=0.5
    evalbot
    evalbot:config=path/to/config.json

Each --policy arg fills the next player slot.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from eval_bot.bot_config import BotConfig
from eval_bot.eval_bot_agent import EvalBotAgent
from game_runner.policy import Policy
from self_play.nn_agent import NNAgent
from training.bc.inference import BCModelHandle, BCPerspective


# Cache loaded model handles so the same checkpoint isn't loaded twice when
# used in multiple slots. Deduped by `BCModelHandle.model_key`, so two specs
# that resolve to the same model (e.g. an arch-bearing checkpoint loaded under
# different — ignored — fallback variants) share one handle and group into one
# batched forward. `_spec_seen` short-circuits a repeated identical spec so it
# doesn't reload just to recompute the key.
_handle_cache: dict[str, BCModelHandle] = {}
_spec_seen: dict[tuple[str, str, str], BCModelHandle] = {}


def _get_or_load_handle(
    path: str, device: torch.device, value_head_variant: str,
) -> BCModelHandle:
    spec_key = (path, str(device), value_head_variant)
    cached = _spec_seen.get(spec_key)
    if cached is not None:
        return cached
    handle = BCModelHandle.load(path, device, value_head_variant)
    handle = _handle_cache.setdefault(handle.model_key, handle)
    _spec_seen[spec_key] = handle
    return handle


def parse_policy_spec(
    spec: str, slot: int, device: torch.device,
) -> Policy:
    parts = spec.split(":", maxsplit=1)
    policy_type = parts[0].lower()
    opts_str = parts[1] if len(parts) > 1 else ""

    if policy_type == "checkpoint":
        return _parse_checkpoint_spec(opts_str, slot, device)
    elif policy_type == "evalbot":
        return _parse_evalbot_spec(opts_str, slot)
    else:
        raise ValueError(
            f"unknown policy type '{policy_type}' in slot {slot}. "
            f"expected 'checkpoint' or 'evalbot'"
        )


def _parse_checkpoint_spec(
    opts_str: str, slot: int, device: torch.device,
) -> NNAgent:
    # First segment is the path, rest are key=value options
    segments = opts_str.split(":")
    if not segments or not segments[0]:
        raise ValueError(
            f"checkpoint policy in slot {slot} requires a path: "
            f"'checkpoint:path/to/model.pt[:key=val,...]'"
        )
    path = segments[0]
    if not Path(path).exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    opts = _parse_kv_opts(":".join(segments[1:])) if len(segments) > 1 else {}

    force_move = _parse_bool(opts.pop("force_move", "false"), "force_move")
    sample = _parse_bool(opts.pop("sample", "false"), "sample")
    temperature = float(opts.pop("temperature", "1.0"))
    value_head_variant = opts.pop("value_head_variant", "direct")

    if opts:
        raise ValueError(
            f"unknown checkpoint options in slot {slot}: {list(opts.keys())}"
        )

    handle = _get_or_load_handle(path, device, value_head_variant)
    perspective = BCPerspective(
        slot,
        device,
        handle.model.cfg.obs,
        force_move=force_move,
        sample=sample,
        temperature=temperature,
    )
    return NNAgent(handle, perspective)


def _parse_evalbot_spec(opts_str: str, slot: int) -> EvalBotAgent:
    opts = _parse_kv_opts(opts_str) if opts_str else {}

    cfg = BotConfig()
    config_path = opts.pop("config", None)
    if config_path is not None:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"evalbot config not found: {config_path}")
        with open(config_path) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(
                    f"unknown BotConfig field '{key}' in evalbot config for slot {slot}"
                )
            setattr(cfg, key, value)

    if opts:
        raise ValueError(
            f"unknown evalbot options in slot {slot}: {list(opts.keys())}"
        )

    return EvalBotAgent(perspective_slot=slot, cfg=cfg)


def _parse_kv_opts(opts_str: str) -> dict[str, str]:
    if not opts_str:
        return {}
    result = {}
    for pair in opts_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            key, value = pair.split("=", maxsplit=1)
            result[key.strip()] = value.strip()
        else:
            result[pair] = "true"
    return result


def _parse_bool(value: str, key: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"invalid boolean value '{value}' for {key}, expected true/false")


def describe_policy(spec: str) -> str:
    """Short human-readable label for a policy spec, for log output."""
    parts = spec.split(":", maxsplit=1)
    policy_type = parts[0].lower()
    if policy_type == "checkpoint":
        return "bc-model"
    elif policy_type == "evalbot":
        return "evalbot"
    return spec


def build_policy_names(specs: list[str]) -> list[str]:
    """Build display names from a list of policy specs, numbering
    duplicates (e.g. two checkpoints become 'bc-model 1', 'bc-model 2').
    Unique types get no number."""
    raw = [describe_policy(s) for s in specs]
    counts: dict[str, int] = {}
    for name in raw:
        counts[name] = counts.get(name, 0) + 1

    seen: dict[str, int] = {}
    result = []
    for name in raw:
        if counts[name] == 1:
            result.append(name)
        else:
            seen[name] = seen.get(name, 0) + 1
            result.append(f"{name} {seen[name]}")
    return result
