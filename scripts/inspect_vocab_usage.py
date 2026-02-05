from __future__ import annotations

import os
from collections import Counter
from functools import reduce
from multiprocessing import Pool
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
SFT_DATASET = "HuggingFaceTB/smol-smoltalk"
SEED = 42
TOP_N = 30
SFT_SAMPLE_LIMIT = 10_000


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
    batch: dict[str, Any], tokenizer: PreTrainedTokenizerBase
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
        truncation=False,
        max_length=None,
        return_attention_mask=False,
    )


def count_tokens_chunk(token_ids_list: list[list[int]]) -> Counter[int]:
    """토큰 ID 리스트의 청크를 처리하여 Counter를 반환합니다."""
    counter: Counter[int] = Counter()
    for ids in token_ids_list:
        counter.update(ids)
    return counter


def parallel_count_tokens(
    token_ids_dataset: list[list[int]],
    num_proc: int,
) -> Counter[int]:
    """병렬 처리로 토큰을 집계합니다."""
    chunk_size = max(1, len(token_ids_dataset) // num_proc)
    chunks = [
        token_ids_dataset[i : i + chunk_size]
        for i in range(0, len(token_ids_dataset), chunk_size)
    ]

    with Pool(processes=num_proc) as pool:
        counters = list(
            tqdm(
                pool.imap(count_tokens_chunk, chunks),
                total=len(chunks),
                desc="📊 토큰 집계 (병렬)",
            )
        )

    return reduce(lambda a, b: a + b, counters, Counter())


def print_report(
    tokenizer: PreTrainedTokenizerBase,
    token_counter: Counter[int],
    top_n: int,
) -> None:
    """토큰 분석 결과를 출력합니다."""
    vocab_size = len(tokenizer)
    unused_tokens = [
        token_id for token_id in range(vocab_size) if token_id not in token_counter
    ]
    low_freq_tokens = token_counter.most_common()[-top_n:][::-1]

    # 미사용 토큰
    print("\n[SFT] 미사용 토큰 목록")
    print(f"COUNT={len(unused_tokens)}")
    print("TOKEN_IDS=" + ", ".join(map(str, unused_tokens[:top_n])))

    # 저빈도 토큰
    print("\n[SFT] 저빈도 토큰 상위 목록")
    for rank, (token_id, count) in enumerate(low_freq_tokens, start=1):
        token_str = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        print(f"{rank:02d}. TOKEN_ID={token_id} | STR='{token_str}' | OCC={count}")

    # 재배치 후보
    print("\n[SFT] 보캡 재배치 후보")
    print("UNUSED_TOP=" + ", ".join(map(str, unused_tokens[:top_n])))
    print("LOW_FREQ_TOP=" + ", ".join(str(tid) for tid, _ in low_freq_tokens))


def main() -> None:
    """CLI 실행 진입점."""
    # 시드 설정
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 토크나이저 로드
    print("📚 토크나이저 로드 중...")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 데이터셋 로드
    print(f"📊 데이터셋 로드 중: {SFT_DATASET}")
    dataset = load_dataset(SFT_DATASET, split="train")

    if SFT_SAMPLE_LIMIT and len(dataset) > SFT_SAMPLE_LIMIT:
        dataset = dataset.select(range(SFT_SAMPLE_LIMIT))
        print(f"✂️  샘플 제한 적용: {SFT_SAMPLE_LIMIT:,}건")

    print(f"📝 총 샘플 수: {len(dataset):,}건")

    # 병렬 토큰화
    num_proc = max(1, (os.cpu_count() or 1) // 2)
    print(f"⚙️  병렬 처리 시작 ({num_proc} 프로세스)...")

    processed_dataset = dataset.map(
        lambda batch: process_batch(batch, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="🔄 토큰화 진행",
    )

    # 토큰 카운팅 (병렬)
    token_counter = parallel_count_tokens(processed_dataset["input_ids"], num_proc)

    # 결과 출력
    print_report(tokenizer, token_counter, TOP_N)


if __name__ == "__main__":
    main()
