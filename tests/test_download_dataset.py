"""다운로드 스크립트 테스트."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class _FakeDataset:
    saved_path: Path | None = None

    def save_to_disk(self, dataset_path: str | Path) -> None:
        self.saved_path = Path(dataset_path)


def _load_module() -> object:
    """스크립트 모듈을 로드합니다."""

    module_path = Path("scripts") / "download_dataset.py"
    spec = importlib.util.spec_from_file_location("download_dataset", module_path)
    if spec is None or spec.loader is None:
        msg = "스크립트 모듈을 로드할 수 없습니다."
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_dataset_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_module()
    fake_dataset = _FakeDataset()

    def _fake_load_dataset(dataset: str, name: str | None = None, split: str | None = None) -> _FakeDataset:
        return fake_dataset

    monkeypatch.setattr(module, "load_dataset", _fake_load_dataset)

    spec = module.DownloadSpec(
        dataset="test-dataset",
        config=None,
        split=None,
        output_dir=tmp_path / "out",
    )
    result_dir = module.download_dataset(spec)

    assert result_dir == tmp_path / "out"
    assert fake_dataset.saved_path == tmp_path / "out"


def test_resolve_output_dir_sanitizes_name() -> None:
    module = _load_module()

    output_dir = module._resolve_output_dir("team/data", None)

    assert output_dir == Path("datasets") / "raw" / "team__data"


def test_validate_dataset_name_rejects_empty() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="비어 있을 수 없습니다"):
        module._validate_dataset_name("   ")
