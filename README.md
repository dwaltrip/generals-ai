# generals-ai

Building an AI bot to play the [generals.io](https://generals.io) strategy game in FFA mode.

## Setup (one-time, after clone)

```sh
./tools/setup-git-hooks.sh
```

Points git at the repo's tracked hooks under `.githooks/`, which includes a pre-commit hook that regenerates `modal_requirements.txt` files whenever `uv.lock` changes.

## Where to look next

- [`AGENTS.md`](./AGENTS.md) — project overview, sub-projects, key entry points, tooling notes.
- [`docs/`](./docs/) — design docs and references (game format, API, network architecture, etc.).
- [`replay-collector/README.md`](./replay-collector/README.md) — operator guide for the replay collector sub-project.
