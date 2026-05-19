# Training Spike 1 — Status

Working guide: [`2026-05/5.17-4-training-spike-plan.md`](./2026-05/5.17-4-training-spike-plan.md). See [§3](./2026-05/5.17-4-training-spike-plan.md#3-the-plan) for the phase-by-phase sketches.

Chunk D sub-step spec: [`2026-05/5.18-3-phase-1d-spec-and-pass-head-audit.md`](./2026-05/5.18-3-phase-1d-spec-and-pass-head-audit.md). All sub-steps shipped; the doc is historical now.

## Progress

### Phase 1 — Environment + dataloader skeleton + cheap probes

- [x] **A** — `training/` subpackage bootstrap, torch 2.12 installed, MPS available
- [x] **B** — corpus pass-rate + map-size sweeps
- [x] **C** — MPS benchmark on a placeholder U-Net
- [x] **D** — dataloader skeleton + smoke test
  - [x] **D1** — action encode/decode + round-trip tests (`bc/actions.py`, `bc/constants.py`, `tests/test_actions.py`)
  - [x] **D2** — per-game iteration scaffold (`bc/dataset.py`, `tests/test_dataset.py`)
  - [x] **D3** — 12-channel obs construction (`bc/obs.py`, `tests/test_obs.py`)
  - [x] **D4** — legality mask + `encode_frame` + raw/encoded iteration split (`bc/mask.py`, `bc/dataset.py`, `tests/test_mask.py`)
  - [x] **D5** — DataLoader integration smoke (`tests/test_dataloader_smoke.py`; 6.4k samples in ~4s)

Full suite: 30 tests green in ~12s.

### Phases 2–4 — not yet started

## Key empirical findings

| Item | Result | Script |
|---|---|---|
| Pass-rate (perspective slots only, full 165k corpus) | **49.7%** corpus-weighted; per-game bimodal (p10=0.11 / p50=0.46 / p90=0.87) | `training/investigations/5_18_pass_rate_sweep.py` |
| Map-size: max(w,h) > 32 fraction | **7.18%**, dominated by max=33 (5.77% of corpus) | `training/investigations/5_18_map_size_sweep.py` |
| MPS speedup over CPU on placeholder U-Net | **6.88×**, no MPS-fallback errors on Conv2d / ConvTranspose2d / GroupNorm / ReLU | `training/scripts/mps_benchmark.py` |

## Resolved decisions

| Item | Choice | Source |
|---|---|---|
| Map padding policy | Pad to 32×32, drop games with max(w,h) > 32 | empirical: 10% corpus trim, ~25–30% compute saved vs. plan's 36×36 |
| Device for spike | MPS | 6.88× faster than CPU on placeholder U-Net, no fallback errors |
| Initial pass-head weight `μ` | Keep at 1.0 for v1 | 49.7% is roughly balanced; revisit if loss imbalance shows up in training |
| Direction enum | NESW clockwise: `N=0, E=1, S=2, W=3` | naturalness; codified in `bc/actions.py` + `5.18-3` |
| Policy-head flat layout | Cell-major: `flat_idx = cell_padded * 8 + sub` | mask construction reads cell-outer naturally; encoding math constant in board dims; model side permutes (~2 MB float copy per batch — negligible) |
| Constants home | `bc/actions.py` (encoding semantics: direction enum + sub/flat layout); `bc/constants.py` (shape + pipeline: `H_PADDED`/`W_PADDED`, padding convention, drop-filter thresholds) | per-concern split — `actions.py` shouldn't own padding or eligibility |
| Package layout | `training/bc/` (renamed from `training/training/`) | clears pytest namespace-package collision from identical workspace/package names; sets up future `training/ppo/` (self-play) + shared module |
| Slot canonicalization ordering | Ascending-skip: channel 0 = perspective, channels 1..7 = remaining slots in ascending raw-slot order with perspective removed | matches non-cyclic literature pattern (Dota/OpenAI Five, AlphaStar, Pluribus); generals.io is simultaneous-action so cyclic ordering (Hanabi/Mahjong) has no game-mechanics anchor here. Full rationale in `bc/obs.canonical_slot_order` docstring |
| Eliminated-perspective frames | Stop walking at `min(T-1, elim_timestep[k])` when elim != -1 | post-elim frames are all-pass and carry no training signal; filtering avoids teaching the model to pass when dead |
| Per-sample dict contract (DataLoader output, post-collate at batch B) | `{"obs": float32 [B, 12, 32, 32], "mask": bool [B, 32, 32, 8], "action_target": int64 [B], "is_pass": bool [B], "value_target": int64 [B]}` | channels-first obs is Conv2d-native; mask shape flattens to `[B, H*W*8]` matching `action_target`'s range; `action_target = -1` for pass (cross_entropy ignore_index idiom) |
| Dataset iteration entry points | `iter_frames()` yields raw `Frame(sim, meta, k, t)` for tests + inspection; `__iter__` yields encoded dicts for `DataLoader` | clean separation: raw walk testable in isolation, production path encodes via `encode_frame` |

## Open / forward-looking notes

- Per-game pass-rate is **bimodal**: active vs. sit-back ("open-turtle") games. `μ` may not be the right knob if some games dominate the loss; consider for Phase 2 loss design.
- `load_replay_months()` in `training/investigations/_helpers.py` reaches into the collector sqlite DB for per-replay timestamps because the meta sidecar lacks a date field. Worth fixing if month-bucketing becomes a recurring need across training-side code; see TODO at top of that file.
- `utils/utils/` has the same workspace/package name collision that bit `training/training/` — latent because no tests target it yet. Resolve before training-side code starts importing from utils in tests.
