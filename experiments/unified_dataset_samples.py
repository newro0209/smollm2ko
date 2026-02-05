"""한국어 데이터셋 3종을 동일 포맷으로 다루는 Lazy Transform 예제를 제공합니다."""

from __future__ import annotations

import random
import textwrap
from typing import Any, Callable

from datasets import Dataset, concatenate_datasets, load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


DATASET_NAMES = {
    "koalpaca": "beomi/KoAlpaca-v1.1a",
    "sharegpt": "dbdu/ShareGPT-74k-ko",
    "ko_qa": "kikikara/ko_QA_dataset",
}
SAMPLE_COUNT = 3
SEED = 42
SYSTEM_PROMPT = "당신은 간결하고 정확한 한국어 어시스턴트입니다."
SOURCE_STYLES = {
    DATASET_NAMES["koalpaca"]: "bright_green",
    DATASET_NAMES["sharegpt"]: "bright_cyan",
    DATASET_NAMES["ko_qa"]: "bright_magenta",
}
ROLE_STYLES = {
    "system": "bold bright_yellow",
    "user": "bold bright_blue",
    "assistant": "bold bright_green",
}
ROLE_LABELS = {
    "system": "[SYS]",
    "user": "[USR]",
    "assistant": "[ASST]",
}


def _to_text_koalpaca(example: dict[str, Any]) -> dict[str, Any]:
    """KoAlpaca 예제를 통일 포맷으로 변환합니다."""

    instruction = str(example.get("instruction", ""))
    output = str(example.get("output", ""))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ],
        "source": DATASET_NAMES["koalpaca"],
    }


def _to_text_sharegpt(example: dict[str, Any]) -> dict[str, Any]:
    """ShareGPT 예제를 통일 포맷으로 변환합니다."""

    conversations = example.get("conversations", [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in conversations:
        role = "user" if turn.get("from") == "human" else "assistant"
        messages.append({"role": role, "content": turn.get("value", "")})
    return {"messages": messages, "source": DATASET_NAMES["sharegpt"]}


def _to_text_ko_qa(example: dict[str, Any]) -> dict[str, Any]:
    """ko_QA_dataset 예제를 통일 포맷으로 변환합니다."""

    question = str(example.get("input", ""))
    answer = str(example.get("output", ""))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "source": DATASET_NAMES["ko_qa"],
    }


def _apply_lazy_transform(
    dataset: Dataset, transform: Callable[[dict[str, Any]], dict[str, Any]]
) -> Dataset:
    """set_transform을 사용해 Lazy Transform을 적용합니다."""

    dataset.set_transform(transform)
    return dataset


def _to_unified_single(example: dict[str, Any]) -> dict[str, Any]:
    """단일 예제를 통일 포맷으로 변환합니다."""

    source = example.get("source")
    if source == DATASET_NAMES["koalpaca"]:
        return _to_text_koalpaca(example)
    if source == DATASET_NAMES["sharegpt"]:
        return _to_text_sharegpt(example)
    if source == DATASET_NAMES["ko_qa"]:
        return _to_text_ko_qa(example)
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "source": str(source) if source is not None else "unknown",
    }


def _to_unified_example(example: dict[str, Any]) -> dict[str, Any]:
    """데이터셋 출처에 따라 통일 포맷으로 변환합니다."""

    source = example.get("source")
    if isinstance(source, list):
        items = []
        for idx in range(len(source)):
            item = {
                key: (value[idx] if isinstance(value, list) else value)
                for key, value in example.items()
            }
            items.append(_to_unified_single(item))
        return {
            "messages": [item["messages"] for item in items],
            "source": [item["source"] for item in items],
        }
    return _to_unified_single(example)


def build_unified_dataset() -> Dataset:
    """세 개 데이터셋을 하나처럼 사용하는 Dataset을 구성합니다."""

    koalpaca = load_dataset(DATASET_NAMES["koalpaca"], split="train")
    sharegpt = load_dataset(DATASET_NAMES["sharegpt"], split="train")
    ko_qa = load_dataset(DATASET_NAMES["ko_qa"], split="train")

    datasets = [
        koalpaca.add_column(
            "source", [DATASET_NAMES["koalpaca"]] * len(koalpaca)
        ),
        sharegpt.add_column(
            "source", [DATASET_NAMES["sharegpt"]] * len(sharegpt)
        ),
        ko_qa.add_column("source", [DATASET_NAMES["ko_qa"]] * len(ko_qa)),
    ]
    unified = concatenate_datasets(datasets)
    return _apply_lazy_transform(unified, _to_unified_example)


def main() -> None:
    """샘플 3개를 출력합니다."""

    unified = build_unified_dataset()
    rng = random.Random(SEED)
    total = len(unified)
    if total == 0:
        return
    sample_size = min(SAMPLE_COUNT, total)
    indices = rng.sample(range(total), k=sample_size)
    console = Console()
    wrap_width = max(40, console.width - 8)
    for idx, dataset_index in enumerate(indices, start=1):
        example = unified[dataset_index]
        source = str(example.get("source", "unknown"))
        header = f"SAMPLE {idx} | SOURCE={source} | INDEX={dataset_index}"
        body = Text()
        for message in example["messages"]:
            role = message["role"]
            content = message["content"]
            role_style = ROLE_STYLES.get(role, "bold")
            role_label = ROLE_LABELS.get(role, role.upper()) or role.upper()
            wrapped = textwrap.wrap(
                content, width=wrap_width - (len(role_label) + 2)
            ) or [""]
            body.append(f"{role_label}: ", style=role_style)
            body.append(f"{wrapped[0]}\n")
            for line in wrapped[1:]:
                body.append(" " * (len(role_label) + 2))
                body.append(f"{line}\n")
        source_style = SOURCE_STYLES.get(source, "white")
        console.print(
            Panel(
                body,
                title=Text(header, style=source_style),
                expand=False,
                border_style=source_style,
            )
        )


if __name__ == "__main__":
    main()
