"""SmolLM2 vocab의 퇴출 후보 토큰(evicted tokens)을 추출하고 TOON 형식으로 출력합니다."""

from __future__ import annotations

import argparse
import logging
import os
import re
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


def _count_tokens_chunk(token_ids_chunk: list[list[int]]) -> Counter[int]:
    """토큰 ID 청크를 카운팅합니다."""
    return Counter(chain.from_iterable(token_ids_chunk))


def _extract_text_from_sample(sample: dict[str, Any], columns: set[str]) -> str:
    """샘플에서 텍스트를 추출합니다."""
    if "text" in columns:
        return str(sample.get("text") or "")

    if "messages" in columns:
        messages = sample.get("messages") or []
        return "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages
        )

    return " ".join(str(value) for value in sample.values())


def _process_batch(
    batch: dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> BatchEncoding:
    """배치 단위로 텍스트를 추출하고 토큰화합니다."""
    columns = set(batch.keys())
    batch_size = len(next(iter(batch.values())))

    texts = [
        _extract_text_from_sample({col: batch[col][i] for col in columns}, columns)
        for i in range(batch_size)
    ]

    return tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )


def _is_prunable_token(token_id: int, tokenizer: PreTrainedTokenizerBase) -> bool:
    """
    토큰이 제거 가능한지(prunable) 판단합니다.

    스페셜 토큰, 순수 숫자, 순수 특수문자, 제어 문자, 바이트 레벨 토큰 등은 제거 불가능합니다.
    """
    # 스페셜 토큰 제외
    if token_id in tokenizer.all_special_ids:
        return False

    # 토큰 디코딩 (str로 명시적 변환)
    decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    token_str = str(decoded) if not isinstance(decoded, str) else decoded

    # 빈 문자열이나 공백만 있는 경우 제외
    if not token_str or token_str.isspace():
        return False

    # 바이트 레벨 토큰 제외 (예: <0x00> ~ <0xFF>)
    if re.match(r"^<0x[0-9A-Fa-f]{2}>$", token_str):
        return False

    # 제어 문자 포함 토큰 제외 (개행, 탭, 캐리지 리턴 등)
    if any(ord(c) < 32 or c in "\n\t\r" for c in token_str):
        return False

    stripped = token_str.strip()

    # 숫자만 있는 토큰 제외
    if stripped and stripped.isdigit():
        return False

    # 특수문자만 있는 토큰 제외 (알파벳, 숫자, 한글이 없는 경우)
    if stripped and not re.search(r"[a-zA-Z0-9가-힣]", stripped):
        return False

    # 공백 마커가 포함된 토큰 제외 (GPT-2의 Ġ, SentencePiece의 ▁ 등)
    if "Ġ" in token_str or "▁" in token_str:
        return False

    # 매우 짧은 고빈도 토큰 제외 (1-2글자 영문/숫자)
    if len(stripped) <= 2 and stripped.isalnum():
        return False

    return True


def main() -> None:
    """CLI 실행 진입점."""
    # ============================================================
    # 1️⃣ 초기 설정 및 로깅 준비
    # ============================================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("extract_evicted_tokens.log", encoding="utf-8")
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("퇴출 후보 토큰 추출 작업 시작")

    disable_progress_bars()

    # ============================================================
    # 2️⃣ CLI 인자 파싱 및 검증
    # ============================================================
    parser = argparse.ArgumentParser(
        description="Extract evicted token candidates for vocabulary adaptation in TOON format"
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
        "--eviction-ratio",
        type=float,
        default=0.3,
        help="퇴출할 토큰의 비율 (0.0 ~ 1.0, 기본값: 0.3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="결과를 저장할 .toon 파일 경로 (지정하지 않으면 stdout 출력)",
    )
    args = parser.parse_args()
    logger.info(
        f"입력 인자: model={args.model_name}, dataset={args.dataset_name}, eviction_ratio={args.eviction_ratio}"
    )

    if not 0.0 <= args.eviction_ratio <= 1.0:
        logger.error(f"잘못된 eviction-ratio 값: {args.eviction_ratio}")
        raise ValueError("eviction-ratio must be between 0.0 and 1.0")

    # ============================================================
    # 3️⃣ 재현성을 위한 시드 설정
    # ============================================================
    logger.info(f"시드 설정: {SEED}")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("CUDA 시드 및 결정론적 모드 활성화")

    # ============================================================
    # 4️⃣ 토크나이저 로드 및 설정
    # ============================================================
    logger.info(f"토크나이저 로드 시작: {args.model_name}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True
    )
    logger.info(f"토크나이저 로드 완료 (vocab_size={len(tokenizer)})")

    # 토크나이저 최대 길이 계산 (한 번만 수행)
    max_length = tokenizer.model_max_length
    if max_length > 100_000:
        try:
            config = AutoConfig.from_pretrained(args.model_name)
            max_length = getattr(config, "max_position_embeddings", max_length)
            logger.info(f"모델 설정에서 최대 길이 조정: {max_length}")
        except Exception:
            logger.warning("모델 설정에서 최대 길이 조정 실패")
    logger.info(f"토큰화 최대 길이: {max_length}")

    # ============================================================
    # 5️⃣ 데이터셋 로드 및 병렬 토큰화
    # ============================================================
    logger.info(f"데이터셋 로드 시작: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    logger.info(f"데이터셋 로드 완료 (샘플 수={len(dataset)})")

    num_proc = max(1, (os.cpu_count() or 1) - 1)
    logger.info(f"병렬 토큰화 시작 (워커 수={num_proc})")

    processed_dataset = dataset.map(
        lambda batch: _process_batch(batch, tokenizer, max_length),
        batched=True,
        batch_size=1000,  # 더 큰 배치 크기로 성능 향상
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        writer_batch_size=1000,  # I/O 최적화
        desc=None,
    )
    logger.info("병렬 토큰화 완료")

    # ============================================================
    # 6️⃣ 토큰 사용 빈도 분석 (병렬 처리)
    # ============================================================
    logger.info("토큰 카운팅 시작")
    token_ids = processed_dataset["input_ids"]
    total_tokens = len(token_ids)
    chunk_size = max(1, total_tokens // num_proc)

    logger.info(f"병렬 카운팅 (프로세스={num_proc}, 청크당={chunk_size}개)")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = [
            executor.submit(_count_tokens_chunk, token_ids[i : i + chunk_size])
            for i in range(0, total_tokens, chunk_size)
        ]
        counters = [f.result() for f in futures]

    token_counter = reduce(lambda a, b: a + b, counters, Counter())
    logger.info(f"토큰 카운팅 완료 (고유 토큰 수={len(token_counter)})")

    # ============================================================
    # 7️⃣ 퇴출 후보 토큰 선정 (미사용 + 저빈도)
    # ============================================================
    logger.info("제거 가능한 토큰 필터링 시작 (스페셜 토큰, 숫자, 특수문자 제외)")
    vocab_size = len(tokenizer)

    # 제거 가능한 모든 토큰 추출 (순차 처리)
    prunable_tokens = [
        token_id
        for token_id in range(vocab_size)
        if _is_prunable_token(token_id, tokenizer)
    ]
    logger.info(f"제거 가능한 전체 토큰 수: {len(prunable_tokens)}")

    # 미사용 토큰 추출 (저효용 토큰)
    unused_tokens = [
        token_id for token_id in prunable_tokens if token_id not in token_counter
    ]
    logger.info(f"미사용 토큰 수: {len(unused_tokens)}")

    num_evicted = max(1, int(len(prunable_tokens) * args.eviction_ratio))
    logger.info(
        f"퇴출 후보 목표 개수: {num_evicted} (비율={args.eviction_ratio})"
    )

    num_unused = min(len(unused_tokens), num_evicted)
    selected_unused = unused_tokens[:num_unused]
    logger.info(f"선택된 미사용 토큰 수: {num_unused}")

    num_low_freq = max(0, num_evicted - num_unused)

    # 저빈도 토큰도 제거 가능한 토큰 중에서만 선택
    prunable_token_set = set(prunable_tokens)
    low_freq_candidates = [
        (tid, count)
        for tid, count in token_counter.most_common()
        if tid in prunable_token_set
    ]
    low_freq_prunable = (
        low_freq_candidates[-num_low_freq:][::-1] if num_low_freq > 0 else []
    )
    logger.info(f"선택된 저빈도 토큰 수: {len(low_freq_prunable)}")

    evicted_tokens = set(selected_unused) | {tid for tid, _ in low_freq_prunable}
    logger.info(f"총 퇴출 예정 토큰 수: {len(evicted_tokens)}")
    logger.info(f"퇴출 토큰 샘플 (최대 5개): {sorted(list(evicted_tokens))[:5]}")

    # ============================================================
    # 8️⃣ TOON 형식으로 결과 변환
    # ============================================================
    logger.info("TOON 형식으로 결과 생성 중")
    result = [
        {
            str(token_id): tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )
        }
        for token_id in sorted(evicted_tokens)  # 정렬하여 일관성 있는 출력
    ]

    output_content = encode(result)
    logger.info("TOON 형식 변환 완료")

    # ============================================================
    # 9️⃣ 결과 저장 또는 출력
    # ============================================================
    if args.output:
        # 파일로 저장
        output_path = (
            args.output if args.output.endswith(".toon") else f"{args.output}.toon"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        logger.info(f"결과 파일 저장 완료: {output_path}")
    else:
        # stdout 출력
        print(output_content)
        logger.info("결과를 stdout으로 출력 완료")

    logger.info("퇴출 후보 토큰 추출 작업 완료")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
