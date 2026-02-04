"""raw 데이터셋 구조 점검 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from datasets import Dataset, DatasetDict, load_from_disk
try:
    from toon_format import encode
except Exception as exc:  # pragma: no cover - 외부 패키지 로드 실패 대비
    encode = None
    _ENCODE_ERROR: Exception | None = exc
else:
    _ENCODE_ERROR = None


def _human_bytes(num_bytes: int) -> str:
    """바이트 수를 사람이 읽기 쉬운 단위로 변환합니다."""

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def _summarize_total_bytes(root: Path) -> int:
    """총 파일 크기를 계산합니다."""

    total_bytes = 0
    for item in root.rglob("*"):
        if item.is_file():
            try:
                total_bytes += item.stat().st_size
            except FileNotFoundError:
                continue
    return total_bytes


def _truncate_text(value: object, max_len: int) -> str:
    """출력 길이를 제한합니다."""

    text = repr(value)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _format_schema(dataset: Dataset) -> list[str]:
    """테이블 구조를 사람이 읽기 쉬운 형태로 변환합니다."""

    lines: list[str] = []
    for name, feature in dataset.features.items():
        lines.append(f"{name}: {feature}")
    if not lines:
        lines.append("컬럼 정보 없음")
    return lines


def _format_sample(dataset: Dataset) -> list[str]:
    """샘플 한 건을 문자열로 변환합니다."""

    if dataset.num_rows == 0:
        return ["샘플 없음"]

    sample = dataset[0]
    lines: list[str] = []
    for key, value in sample.items():
        lines.append(f"{key}: {_truncate_text(value, 160)}")
    if not lines:
        lines.append("샘플 없음")
    return lines


def _select_dataset(dataset_obj: Dataset | DatasetDict) -> tuple[list[str], Dataset | None]:
    """split 목록과 출력용 Dataset을 선택합니다."""

    if isinstance(dataset_obj, DatasetDict):
        splits = sorted(dataset_obj.keys())
        if splits:
            return splits, dataset_obj[splits[0]]
        return [], None

    if isinstance(dataset_obj, Dataset):
        return ["single"], dataset_obj

    return [], None


def _summarize_dataset(
    dataset_dir: Path,
) -> tuple[Path, int, list[str] | None, list[str], list[str]]:
    """데이터셋 디렉터리의 요약 정보를 생성합니다."""

    total_bytes = _summarize_total_bytes(dataset_dir)
    load_error: Exception | None = None
    dataset_obj: Dataset | DatasetDict | None = None

    try:
        dataset_obj = load_from_disk(str(dataset_dir))
    except Exception as exc:
        load_error = exc

    splits: list[str] | None
    preview_dataset: Dataset | None
    if dataset_obj is None:
        splits = None
        preview_dataset = None
    else:
        splits, preview_dataset = _select_dataset(dataset_obj)

    if preview_dataset is None:
        error_text = "로드 실패" if load_error is None else f"로드 실패: {load_error}"
        schema_lines = [error_text]
        sample_lines = [error_text]
        splits = None if load_error is not None else splits
    else:
        schema_lines = _format_schema(preview_dataset)
        sample_lines = _format_sample(preview_dataset)

    return dataset_dir, total_bytes, splits, schema_lines, sample_lines


def _build_dataset_record(
    path: Path,
    total_bytes: int,
    splits: list[str] | None,
    schema_lines: list[str],
    sample_lines: list[str],
) -> dict[str, object]:
    """TOON 출력용 데이터셋 레코드를 구성합니다."""

    return {
        "path": str(path),
        "totalSize": _human_bytes(total_bytes),
        "splits": splits,
        "schema": schema_lines,
        "sample": sample_lines,
    }


def _require_encoder() -> Callable[[object], str]:
    """TOON 인코더를 준비합니다."""

    if encode is None:
        detail = "" if _ENCODE_ERROR is None else f" ({_ENCODE_ERROR})"
        message = (
            "오류: toon_format을 불러올 수 없습니다."
            f"{detail}\\n"
            "설치 방법: pip install git+https://github.com/toon-format/toon-python.git"
        )
        raise RuntimeError(message)
    return encode


def _build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 구성합니다."""

    parser = argparse.ArgumentParser(description="raw 데이터셋 구조 점검")
    parser.add_argument(
        "--root",
        default="datasets/raw",
        help="raw 데이터셋 루트 경로",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="출력할 데이터셋 개수 제한",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 엔트리 포인트."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"오류: 경로가 존재하지 않습니다: {root}", file=sys.stderr)
        return 2

    dataset_dirs = [item for item in sorted(root.iterdir()) if item.is_dir()]
    if args.limit is not None:
        dataset_dirs = dataset_dirs[: args.limit]

    try:
        encoder = _require_encoder()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    dataset_records = [
        _build_dataset_record(*_summarize_dataset(dataset_dir)) for dataset_dir in dataset_dirs
    ]
    output = encoder({"datasets": dataset_records})
    print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
