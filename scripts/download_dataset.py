"""허깅페이스 데이터셋 다운로드 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset


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


def _resolve_output_dir(dataset: str) -> Path:
    """출력 경로를 결정합니다."""

    safe_name = _validate_dataset_name(dataset)
    return Path("datasets") / "raw" / safe_name.split("__")[-1]


def download_dataset(dataset: str) -> Path:
    """허깅페이스 데이터셋을 다운로드하고 디스크에 저장합니다."""

    target_dir = _resolve_output_dir(dataset)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_obj = load_dataset(dataset)

    dataset_obj.save_to_disk(target_dir)
    return target_dir


def _build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 구성합니다."""

    parser = argparse.ArgumentParser(description="허깅페이스 데이터셋 다운로드")
    parser.add_argument("dataset", help="허깅페이스 데이터셋 이름")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 엔트리 포인트."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result_dir = download_dataset(args.dataset)
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
