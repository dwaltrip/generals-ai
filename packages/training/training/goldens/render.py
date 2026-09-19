"""One-line renderings of the comparison records, for test failures and abort messages."""

from __future__ import annotations

from training.goldens.compare import KeyDiff, KeysetDiff, ObsMismatch, SupervisionMismatch


def render_keyset_diff(d: KeysetDiff) -> str:
    return f"emitted keys differ. extra: {set(d.extra)}, missing: {set(d.missing)}"


def render_key_diff(d: KeyDiff) -> str:
    if d.changed_rows is None:
        return (
            f"{d.key}: shape/dtype {d.got.shape} {d.got.dtype}"
            f" vs ref {d.ref.shape} {d.ref.dtype}"
        )
    return (
        f"{d.key}: {d.changed_rows.size} of {d.ref.shape[0]} rows changed,"
        f" first t={int(d.changed_rows[0])}"
    )


def render_obs_mismatch(d: ObsMismatch) -> str:
    parts = []
    if d.changed_frames is None:
        parts.append(f"frame count {d.ref.n_frames} -> {d.got.n_frames}")
    elif d.changed_frames.size:
        parts.append(
            f"{d.changed_frames.size} of {d.ref.n_frames} frames changed"
            f" (first t={int(d.changed_frames[0])})"
        )
    if d.changed_channels is None:
        parts.append(f"channel count {d.ref.n_channels} -> {d.got.n_channels}")
    elif d.changed_channels.size:
        parts.append(f"channels {d.changed_channels.tolist()}")
    return "obs mismatch: " + ", ".join(parts)


def render_supervision_mismatch(d: SupervisionMismatch) -> str:
    if d.keyset:
        return f"supervision mismatch: {render_keyset_diff(d.keyset)}"
    lines = [f"supervision mismatch: keys {[k.key for k in d.diffs]}"]
    return "\n".join(lines + [f"  {render_key_diff(k)}" for k in d.diffs])
