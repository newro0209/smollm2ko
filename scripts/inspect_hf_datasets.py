"""허깅페이스 데이터셋 다운로드 전 구조 점검 스크립트.

사용법:
- 단일/복수 이름: python scripts/inspect_hf_datasets.py name1 name2
- 목록 파일: python scripts/inspect_hf_datasets.py --list names.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    NamedSplit,
    load_dataset,
)

from toon_format import encode

DatasetLike = Dataset | IterableDataset
DatasetGroup = DatasetDict | IterableDatasetDict | DatasetLike


def _format_schema(dataset: DatasetLike) -> list[str]:
    """테이블 구조를 사람이 읽기 쉬운 형태로 변환합니다."""

    features = dataset.features
    return (
        [f"{name}: {feature}" for name, feature in features.items()]
        if features
        else ["컬럼 정보 없음"]
    )


def _format_sample(dataset: DatasetLike) -> list[str]:
    """샘플 한 건을 문자열로 변환합니다."""

    try:
        sample = next(iter(dataset))
    except StopIteration:
        return ["샘플 없음"]
    except Exception as exc:
        return [f"샘플 로드 실패: {exc}"]

    return [f"{key}: {value!r}" for key, value in sample.items()] or ["샘플 없음"]


def _select_dataset(dataset: DatasetGroup) -> tuple[list[str | NamedSplit], DatasetLike | None]:
    """split 목록과 출력용 Dataset을 선택합니다."""

    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        splits = sorted(dataset.keys())
        return splits, dataset[splits[0]] if splits else None

    return ["single"], dataset


def _error_summary(
    name: str,
    error: Exception | None,
    splits: list[str | NamedSplit] | None = None,
) -> tuple[str, list[str | NamedSplit] | None, list[str], list[str]]:
    """로드 실패 정보를 요약합니다."""

    message = "로드 실패" if error is None else f"로드 실패: {error}"
    if error is not None:
        print(f"오류: {error}", file=sys.stderr)
        splits = None
    return name, splits, [message], [message]


def _summarize_dataset(
    name: str,
) -> tuple[str, list[str | NamedSplit] | None, list[str], list[str]]:
    """허깅페이스 데이터셋 정보를 요약합니다."""

    try:
        dataset = load_dataset(name)
    except Exception as exc:
        return _error_summary(name, exc)

    splits, preview = _select_dataset(dataset)
    if preview is None:
        return _error_summary(name, None, splits)

    return (
        name,
        splits,
        _format_schema(preview),
        _format_sample(preview),
    )


def _load_dataset_list(path_str: str) -> list[str]:
    """파일에서 데이터셋 목록을 읽습니다."""

    path = Path(path_str)
    if not path.exists():
        msg = f"목록 파일을 찾을 수 없습니다: {path}"
        raise FileNotFoundError(msg)

    return [
        cleaned
        for line in path.read_text(encoding="utf-8").splitlines()
        if (cleaned := line.strip()) and not cleaned.startswith("#")
    ]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """중복을 제거하면서 입력 순서를 유지합니다."""

    return list(dict.fromkeys(items))


def _build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 구성합니다."""

    parser = argparse.ArgumentParser(description="허깅페이스 데이터셋 다운로드 전 구조 점검")
    parser.add_argument("name", nargs="*", help="허깅페이스 데이터셋 이름")
    parser.add_argument(
        "--list",
        dest="names",
        help="데이터셋 목록 파일(한 줄에 하나)",
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 엔트리 포인트."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    names = list(args.name)
    if args.names:
        try:
            names.extend(_load_dataset_list(args.names))
        except FileNotFoundError as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 2

    names = _dedupe_preserve_order(names)

    if not names:
        print("오류: 데이터셋 이름을 하나 이상 지정해야 합니다.", file=sys.stderr)
        return 2

    dataset_records = [
        {
            "name": name,
            "splits": splits,
            "schema": schema_lines,
            "sample": sample_lines,
        }
        for name in names
        for name, splits, schema_lines, sample_lines in [_summarize_dataset(name)]
    ]

    output = encode({"datasets": dataset_records})
    print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
