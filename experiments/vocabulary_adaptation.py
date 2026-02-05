"""SmolLM2 Instruct vocab의 한국어 비효율 토큰을 점수화합니다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET_NAMES = [
    "beomi/KoAlpaca-v1.1a",
    "dbdu/ShareGPT-74k-ko",
    "kikikara/ko_QA_dataset",
]
EVAL_PROMPT_KO = "안녕하세요. 오늘의 날씨는 어때요?"
EVAL_PROMPT_EN = "Hello. How is the weather today?"
MAX_NEW_TOKENS = 50
SEED = 42
TOP_N = 30


def transform_koalpaca_batch(batch: dict[str, Sequence[str]]) -> dict[str, list[str]]:
    """KoAlpaca 배치 예제를 통일된 텍스트 필드로 변환합니다."""

    instructions = batch.get("instruction", [])
    outputs = batch.get("output", [])
    return {
        "text": [
            f"사용자: {instruction}\n어시스턴트: {output}"
            for instruction, output in zip(instructions, outputs, strict=True)
        ]
    }


def transform_sharegpt_batch(
    batch: dict[str, Sequence[Sequence[dict[str, str]]]]
) -> dict[str, list[str]]:
    """ShareGPT 배치 예제를 대화 텍스트로 변환합니다."""

    conversations_batch = batch.get("conversations", [])
    texts: list[str] = []
    for conversations in conversations_batch:
        lines = [
            f"{'사용자' if turn.get('from') == 'human' else '어시스턴트'}: {turn.get('value', '')}"
            for turn in conversations
        ]
        texts.append("\n".join(lines))
    return {"text": texts}


def transform_ko_qa_batch(batch: dict[str, Sequence[str]]) -> dict[str, list[str]]:
    """ko_QA_dataset 배치 예제를 QA 텍스트로 변환합니다."""

    inputs = batch.get("input", [])
    outputs = batch.get("output", [])
    return {
        "text": [
            f"질문: {question}\n답변: {answer}"
            for question, answer in zip(inputs, outputs, strict=True)
        ]
    }


def _resolve_device():
    """가용한 디바이스를 선택합니다."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    """재현성을 위해 시드를 고정합니다."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_korean_char(char: str) -> bool:
    """문자가 한글(가/힣) 범위인지 확인합니다."""

    return "가" <= char <= "힣"


def _count_korean_chars(text: str) -> int:
    """문자열 내 한글 글자 수를 계산합니다."""

    return sum(1 for ch in text if _is_korean_char(ch))


@dataclass(frozen=True)
class TokenStat:
    """토큰별 누적 통계를 저장합니다."""

    token_id: int
    token_str: str
    occurrences: int
    total_chars: int
    total_korean_chars: int

    @property
    def inefficiency_score(self) -> float:
        """한국어 대비 비효율 점수를 계산합니다."""

        return self.occurrences / max(1, self.total_korean_chars)


def _accumulate_stats(
    tokenizer: PreTrainedTokenizerBase, texts: Iterable[str]
) -> dict[int, TokenStat]:
    """한국어 텍스트에서 사용된 토큰 통계를 누적합니다."""

    stats: dict[int, TokenStat] = {}
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        for token_id in encoded:
            token_str = cast(
                str,
                tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            )
            total_chars = len(token_str)
            total_korean_chars = _count_korean_chars(token_str)
            if token_id in stats:
                current = stats[token_id]
                stats[token_id] = TokenStat(
                    token_id=token_id,
                    token_str=current.token_str,
                    occurrences=current.occurrences + 1,
                    total_chars=current.total_chars + total_chars,
                    total_korean_chars=current.total_korean_chars + total_korean_chars,
                )
            else:
                stats[token_id] = TokenStat(
                    token_id=token_id,
                    token_str=token_str,
                    occurrences=1,
                    total_chars=total_chars,
                    total_korean_chars=total_korean_chars,
                )
    return stats


def score_korean_inefficiency(
    tokenizer: PreTrainedTokenizerBase, texts: Iterable[str], top_n: int
) -> list[TokenStat]:
    """한국어에서 비효율적인 토큰을 점수화해 상위 결과를 반환합니다."""

    stats = _accumulate_stats(tokenizer, texts)
    return sorted(
        stats.values(), key=lambda stat: stat.inefficiency_score, reverse=True
    )[:top_n]


def run_example(prompts: list[str]) -> list[str]:
    """모델을 로드하고 텍스트를 생성합니다."""

    device = _resolve_device()
    _seed_everything(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(cast(Any, device))

    input_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def _print_korean_inefficiency(stats: list[TokenStat]) -> None:
    """비효율 토큰 점수 결과를 출력합니다."""

    print("\n[KO] 비효율 토큰 점수 상위 목록")
    for rank, stat in enumerate(stats, start=1):
        print(
            f"{rank:02d}. TOKEN_ID={stat.token_id} | "
            f"STR='{stat.token_str}' | "
            f"SCORE={stat.inefficiency_score:.3f} | "
            f"OCC={stat.occurrences} | "
            f"KOR_CHARS={stat.total_korean_chars} | "
            f"TOTAL_CHARS={stat.total_chars}"
        )


def main() -> None:
    """CLI 실행 진입점."""

    prompts = [("KO", EVAL_PROMPT_KO), ("EN", EVAL_PROMPT_EN)]
    prompt_texts = [prompt for _, prompt in prompts]
    results = run_example(prompt_texts)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    ko_stats = score_korean_inefficiency(tokenizer, [EVAL_PROMPT_KO], TOP_N)
    for (label, prompt), result in zip(prompts, results, strict=True):
        print(f"[{label}] PROMPT: {prompt}")
        print(f"[{label}] OUTPUT: {result}")
        print(f"[{label}] OUTPUT_CHARS: {len(result)}")
    _print_korean_inefficiency(ko_stats)


if __name__ == "__main__":
    main()
