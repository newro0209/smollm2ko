from typing import Any


SYSTEM_PROMPT = "당신은 간결하고 정확한 한국어 어시스턴트입니다."

EVAL_PROMPT_KO = "안녕하세요. 오늘의 날씨는 어때요?"
EVAL_PROMPT_EN = "Hello. How is the weather today?"

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
        "source": "beomi/KoAlpaca-v1.1a",
    }


def _to_text_sharegpt(example: dict[str, Any]) -> dict[str, Any]:
    """ShareGPT 예제를 통일 포맷으로 변환합니다."""

    conversations = example.get("conversations", [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in conversations:
        role = "user" if turn.get("from") == "human" else "assistant"
        messages.append({"role": role, "content": turn.get("value", "")})
    return {"messages": messages, "source": "dbdu/ShareGPT-74k-ko"}


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
        "source": "kikikara/ko_QA_dataset",
    }
def _to_text_koalpaca(example: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    """KoAlpaca 예제를 통일 포맷으로 변환합니다."""

    instruction = str(example.get("instruction", ""))
    output = str(example.get("output", ""))
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ],
        "source": "beomi/KoAlpaca-v1.1a",
    }


def _to_text_sharegpt(example: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    """ShareGPT 예제를 통일 포맷으로 변환합니다."""

    conversations = example.get("conversations", [])
    messages = [{"role": "system", "content": system_prompt}]
    for turn in conversations:
        role = "user" if turn.get("from") == "human" else "assistant"
        messages.append({"role": role, "content": turn.get("value", "")})
    return {"messages": messages, "source": "dbdu/ShareGPT-74k-ko"}


def _to_text_ko_qa(example: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    """ko_QA_dataset 예제를 통일 포맷으로 변환합니다."""

    question = str(example.get("input", ""))
    answer = str(example.get("output", ""))
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "source": "kikikara/ko_QA_dataset",
    }
