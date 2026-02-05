"""SmolLM2 vocab의 교체 후보 토큰을 추출하고 TOON 형식으로 출력합니다."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from itertools import chain
from typing import Any

import torch
from datasets import disable_progress_bars, load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)

from toon_format import encode

SEED = 42


def extract_text_from_sample(sample: dict[str, Any], columns: set[str]) -> str:
    """샘플에서 텍스트를 추출합니다."""
    if "text" in columns:
        return str(sample.get("text") or "")

    if "messages" in columns:
        messages = sample.get("messages") or []
        return "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages
        )

    return " ".join(str(value) for value in sample.values())


def process_batch(
    batch: dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> BatchEncoding:
    """배치 단위로 텍스트를 추출하고 토큰화합니다."""
    columns = set(batch.keys())
    batch_size = len(next(iter(batch.values())))

    texts = [
        extract_text_from_sample({col: batch[col][i] for col in columns}, columns)
        for i in range(batch_size)
    ]

    return tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )


def count_tokens_chunk(token_ids_list: list[list[int]]) -> Counter[int]:
    """토큰 ID 리스트의 청크를 처리하여 Counter를 반환합니다."""
    # itertools.chain으로 flatten 후 한 번에 Counter 생성
    return Counter(chain.from_iterable(token_ids_list))


def main() -> None:
    """CLI 실행 진입점."""
    # 진행표시줄 비활성화
    disable_progress_bars()

    parser = argparse.ArgumentParser(
        description="Extract vocabulary replacement candidates in TOON format"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Hugging Face 모델 이름 (e.g., HuggingFaceTB/SmolLM2-135M-Instruct)",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Hugging Face 데이터셋 이름 (e.g., HuggingFaceTB/smol-smoltalk)",
        default="HuggingFaceTB/smol-smoltalk",
    )
    parser.add_argument(
        "--replacement-ratio",
        type=float,
        default=0.3,
        help="교체할 토큰의 비율 (0.0 ~ 1.0, 기본값: 0.3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="결과를 저장할 .toon 파일 경로 (지정하지 않으면 stdout 출력)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.replacement_ratio <= 1.0:
        raise ValueError("replacement-ratio must be between 0.0 and 1.0")

    # 시드 설정
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 토크나이저 로드 (Fast 토크나이저 사용)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True
    )

    # 토크나이저 최대 길이 계산 (한 번만 수행)
    max_length = tokenizer.model_max_length
    if max_length > 100_000:
        try:
            config = AutoConfig.from_pretrained(args.model_name)
            max_length = getattr(config, "max_position_embeddings", max_length)
        except Exception:
            pass

    # 데이터셋 로드
    dataset = load_dataset(args.dataset_name, split="train")

    # 병렬 토큰화
    num_proc = max(1, (os.cpu_count() or 1) - 1)

    processed_dataset = dataset.map(
        lambda batch: process_batch(batch, tokenizer, max_length),
        batched=True,
        batch_size=1000,  # 더 큰 배치 크기로 성능 향상
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        writer_batch_size=1000,  # I/O 최적화
        desc=None,
    )

    # 토큰 카운팅 (병렬)
    token_ids = processed_dataset["input_ids"]
    chunk_size = len(token_ids) // num_proc + 1

    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = [
            executor.submit(
                count_tokens_chunk, token_ids[i : i + chunk_size]
            )
            for i in range(0, len(token_ids), chunk_size)
        ]
        counters = [f.result() for f in futures]

    token_counter = reduce(lambda a, b: a + b, counters, Counter())

    # 미사용 토큰 추출 (스페셜 토큰 피하기 위해 1000번부터)
    vocab_size = len(tokenizer)
    unused_tokens = [
        token_id
        for token_id in range(1000, vocab_size)
        if token_id not in token_counter
    ]

    # 교체 비율에 따라 전체 후보 개수 결정 (미사용 + 저빈도)
    total_candidates = max(1, int((vocab_size - 1000) * args.replacement_ratio))

    # 미사용 토큰 먼저 사용
    num_unused = min(len(unused_tokens), total_candidates)
    selected_unused = unused_tokens[:num_unused]

    # 남은 개수만큼 저빈도 토큰 추출
    num_low_freq = max(0, total_candidates - num_unused)
    low_freq_tokens = (
        token_counter.most_common()[-num_low_freq:][::-1] if num_low_freq > 0 else []
    )

    # 모든 토큰 ID 합치기
    all_token_ids = set(selected_unused) | {tid for tid, _ in low_freq_tokens}

    # 결과 매핑 생성 (한 번만 디코딩)
    result = [
        {
            str(token_id): tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )
        }
        for token_id in all_token_ids
    ]

    output_content = encode(result)

    if args.output:
        # 파일로 저장
        output_path = (
            args.output if args.output.endswith(".toon") else f"{args.output}.toon"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
    else:
        # stdout 출력
        print(output_content)


if __name__ == "__main__":
    main()
