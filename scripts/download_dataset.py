"""허깅페이스 데이터셋 다운로드 스크립트."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from os import PathLike
from typing import Protocol

from datasets import load_dataset


class SaveToDisk(Protocol):
    """save_to_disk를 제공하는 객체 프로토콜."""

    def save_to_disk(  # pragma: no cover - Dataset/DatasetDict 시그니처 차이를 흡수
        self,
        dataset_path: str | Path | PathLike[str] | PathLike[bytes],
        /,
        max_shard_size: str | int | None = None,
        num_shards: None = None,
        num_proc: int | None = None,
        storage_options: dict[object, object] | None = None,
    ) -> None:
        ...


@dataclass(frozen=True)
class DownloadSpec:
    """다운로드 명세."""

    dataset: str
    config: str | None
    split: str | None
    output_dir: Path


def _sanitize_dataset_name(dataset: str) -> str:
    """경로 안전성을 위해 데이터셋 이름을 정제합니다."""

    return dataset.strip().replace("/", "__")


def _validate_dataset_name(dataset: str) -> str:
    """데이터셋 이름을 검증합니다."""

    sanitized = _sanitize_dataset_name(dataset)
    if not sanitized:
        msg = "데이터셋 이름은 비어 있을 수 없습니다."
        raise ValueError(msg)
    return sanitized


def _resolve_output_dir(dataset: str, output_dir: str | None) -> Path:
    """출력 경로를 결정합니다."""

    if output_dir is not None:
        return Path(output_dir)

    safe_name = _validate_dataset_name(dataset)
    return Path("datasets") / "raw" / safe_name


def download_dataset(spec: DownloadSpec) -> Path:
    """허깅페이스 데이터셋을 다운로드하고 디스크에 저장합니다."""

    target_dir = spec.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset: SaveToDisk
    if spec.split is None:
        dataset = load_dataset(spec.dataset, name=spec.config)
    else:
        dataset = load_dataset(spec.dataset, name=spec.config, split=spec.split)

    dataset.save_to_disk(target_dir)
    return target_dir


def _build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 구성합니다."""

    parser = argparse.ArgumentParser(description="허깅페이스 데이터셋 다운로드")
    parser.add_argument("dataset", help="허깅페이스 데이터셋 이름")
    parser.add_argument("--config", help="데이터셋 구성 이름", default=None)
    parser.add_argument("--split", help="다운로드할 split 이름", default=None)
    parser.add_argument(
        "--output-dir",
        help="저장 경로 (기본값: datasets/raw/<dataset_name>)",
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 엔트리 포인트."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output_dir = _resolve_output_dir(args.dataset, args.output_dir)
        spec = DownloadSpec(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            output_dir=output_dir,
        )
        result_dir = download_dataset(spec)
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - 외부 라이브러리 오류 래핑
        print(f"다운로드 실패: {exc}", file=sys.stderr)
        return 1

    print(f"저장 완료: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
