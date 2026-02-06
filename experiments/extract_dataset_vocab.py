"""원하는 데이터셋에서 토크나이저 기준 보캡 사용량을 추출하는 실험 모듈 초안."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TypeAlias

from datasets import Dataset, IterableDataset, load_dataset
from toon_format import encode
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DatasetLike: TypeAlias = Dataset | IterableDataset


@dataclass(slots=True)
class VocabExtractionConfig:
    """보캡 추출 실험 설정입니다."""

    dataset_name: str
    dataset_split: str
    model_name: str
    text_columns: tuple[str, ...]
    sample_limit: int | None
    top_k: int


def _build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 구성합니다."""

    parser = argparse.ArgumentParser(description="데이터셋 기반 보캡 사용량 추출")
    parser.add_argument("--dataset", required=True, help="허깅페이스 데이터셋 이름")
    parser.add_argument("--split", default="train", help="사용할 split 이름")
    parser.add_argument(
        "--model",
        required=True,
        help="토크나이저를 불러올 모델 이름 (예: HuggingFaceTB/SmolLM2-135M-Instruct)",
    )
    parser.add_argument(
        "--text-columns",
        nargs="*",
        default=["text", "messages", "conversations", "instruction", "output"],
        help="텍스트 추출 우선 컬럼 목록",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="샘플 상한 (미지정 시 전체 사용)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="빈도 상위 토큰 개수",
    )
    return parser


def _normalize_message_like(value: Any) -> str:
    """messages/conversations 형태를 문자열로 정규화합니다."""

    if not isinstance(value, list):
        return str(value)

    lines: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            lines.append(str(item))
            continue
        role = str(item.get("role") or item.get("from") or "unknown")
        content = str(item.get("content") or item.get("value") or "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_text(sample: dict[str, Any], text_columns: Sequence[str]) -> str:
    """샘플에서 토큰화 가능한 텍스트를 추출합니다."""

    values: list[str] = []
    for column in text_columns:
        if column not in sample:
            continue
        value = sample[column]
        if column in {"messages", "conversations"}:
            values.append(_normalize_message_like(value))
        else:
            values.append(str(value))

    if values:
        return "\n".join(values)

    return " ".join(str(value) for value in sample.values())


def _iter_samples(dataset: DatasetLike, limit: int | None) -> Iterable[dict[str, Any]]:
    """데이터셋 샘플을 순차적으로 반환합니다."""

    for index, sample in enumerate(dataset):
        if limit is not None and index >= limit:
            return
        yield sample


def _count_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetLike,
    text_columns: Sequence[str],
    limit: int | None,
) -> Counter[int]:
    """데이터셋 전체에서 토큰 ID 빈도를 집계합니다."""

    counter: Counter[int] = Counter()
    for sample in _iter_samples(dataset, limit=limit):
        text = _extract_text(sample, text_columns=text_columns)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(token_ids)
    return counter


def _build_output(
    tokenizer: PreTrainedTokenizerBase,
    config: VocabExtractionConfig,
    token_counter: Counter[int],
) -> dict[str, Any]:
    """출력용 구조를 구성합니다."""

    top_items = token_counter.most_common(config.top_k)
    unique_token_count = len(token_counter)
    total_token_count = sum(token_counter.values())

    return {
        "config": {
            "dataset": config.dataset_name,
            "split": config.dataset_split,
            "model": config.model_name,
            "text_columns": list(config.text_columns),
            "sample_limit": config.sample_limit,
            "top_k": config.top_k,
        },
        "summary": {
            "unique_token_count": unique_token_count,
            "total_token_count": total_token_count,
        },
        "top_tokens": [
            {
                "token_id": token_id,
                "token": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                "count": count,
            }
            for token_id, count in top_items
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 엔트리 포인트."""

    args = _build_parser().parse_args(argv)
    config = VocabExtractionConfig(
        dataset_name=args.dataset,
        dataset_split=args.split,
        model_name=args.model,
        text_columns=tuple(args.text_columns),
        sample_limit=args.limit,
        top_k=args.top_k,
    )

    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    token_counter = _count_token_ids(
        tokenizer=tokenizer,
        dataset=dataset,
        text_columns=config.text_columns,
        limit=config.sample_limit,
    )

    print(encode(_build_output(tokenizer, config, token_counter)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
