"""The cache folder a worker's safety weights live in is fixed by the worker cache root, not by HF_HOME."""

import os
from pathlib import Path

import pytest

import horde_safety


def test_cache_root_pins_clip_blip_even_with_hf_home_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "models"
    monkeypatch.setenv("AIWORKER_CACHE_HOME", str(cache_root))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    assert horde_safety.setup_cache_paths() == str(cache_root / "clip_blip")
    assert os.environ["HF_HOME"] == str(tmp_path / "elsewhere"), "an explicit HF_HOME is left alone"


def test_cache_root_supplies_hf_home_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "models"
    monkeypatch.setenv("AIWORKER_CACHE_HOME", str(cache_root))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    assert horde_safety.setup_cache_paths() == str(cache_root / "clip_blip")
    assert os.environ["HF_HOME"] == str(cache_root / "hf_transformers")


def test_without_a_cache_root_hf_home_decides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    assert horde_safety.setup_cache_paths() == str(tmp_path / "hf")
