"""Shared helpers: load the worker, supervisor and viewer as modules without a GPU stack.

worker.py imports torch at module level. The functions tested here never touch it, so when torch
is not installed a minimal stand-in is registered first (CPU only, no tensors).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from pytest import fixture

ROOT = Path(__file__).resolve().parent.parent


def _torch_stub() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, memory_allocated=lambda: 0)
    torch.nn = types.SimpleNamespace(Module=object)
    torch.Tensor = object
    return torch


def load(name: str, rel: str, argv: list[str] | None = None):
    """Import ROOT/rel as a fresh module called `name`, with sys.argv set for modules that read it on import."""
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.modules["torch"] = _torch_stub()
    old = sys.argv
    sys.argv = argv or [rel]
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


@fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setenv("YGO_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("YGO_DECK", raising=False)
    return load("ygo_worker", "worker/worker.py", ["worker.py", "test"])
