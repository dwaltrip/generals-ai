from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.goldens.lib import store
from training.goldens.lib.cli import regen_main
from training.goldens.lib.content import EntryContent
from training.goldens.lib.registry import Point, Registry, Surface


@dataclass(frozen=True)
class Fx:
    id: str


def _produce(cfg: int, data: np.ndarray) -> EntryContent:
    return {"scaled": data * cfg}


_REGISTRY = Registry(
    surfaces=(Surface(name="s", points=(Point(name="p", cfg=2),), produce=_produce),),
    fixtures=(Fx(id="f"),),
    load_fixture=lambda fx: np.arange(4),
)


def _describe(fx: Fx, data: np.ndarray) -> str:
    return f"n={len(data)}"


def test_regen_main(tmp_path: Path) -> None:
    assert regen_main(_REGISTRY, tmp_path, describe_fixture=_describe, argv=["--dry-run"]) == 0
    assert not store.load_references(tmp_path, {"s"}).references

    assert regen_main(_REGISTRY, tmp_path, describe_fixture=_describe, argv=[]) == 0
    assert len(store.load_references(tmp_path, {"s"}).references) == 1
