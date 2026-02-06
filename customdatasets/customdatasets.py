"""프로젝트 공통 한국어 데이터셋 3종을 통합하는 허깅페이스 데이터셋 빌더."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import datasets

_SYSTEM_PROMPT = "당신은 간결하고 정확한 한국어 어시스턴트입니다."


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """원본 데이터셋 로딩에 필요한 정보를 담는 사양입니다."""

    name: str
    split: str


_SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(name="beomi/KoAlpaca-v1.1a", split="train"),
    SourceSpec(name="dbdu/ShareGPT-74k-ko", split="train"),
    SourceSpec(name="kikikara/ko_QA_dataset", split="train"),
)


class CustomDatasets(datasets.GeneratorBasedBuilder):
    """3개 한국어 데이터셋을 단일 스키마로 합치는 빌더입니다."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="KoAlpaca, ShareGPT-74k-ko, ko_QA_dataset 통합",
        )
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> datasets.DatasetInfo:
        """통합 데이터셋 스키마를 정의합니다."""

        return datasets.DatasetInfo(
            description="이 프로젝트 실험용 한국어 통합 데이터셋",
            features=datasets.Features(
                {
                    "messages": datasets.Sequence(
                        feature={
                            "role": datasets.Value("string"),
                            "content": datasets.Value("string"),
                        }
                    ),
                    "source": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        """외부 데이터셋을 다운로드하고 train split 생성을 정의합니다."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"dl_manager": dl_manager, "source_specs": _SOURCE_SPECS},
            )
        ]

    def _generate_examples(
        self,
        dl_manager: datasets.DownloadManager,
        source_specs: Iterable[SourceSpec],
    ) -> Iterable[tuple[str, dict[str, Any]]]:
        """원본 데이터셋을 순회하며 통합 포맷 예제를 생성합니다."""

        idx = 0
        for spec in source_specs:
            dataset = datasets.load_dataset(
                spec.name,
                split=spec.split,
                download_config=dl_manager.download_config,
            )

            for sample in dataset:
                example = self._to_unified_example(spec.name, sample)
                yield f"{spec.name}-{idx}", example
                idx += 1

    def _to_unified_example(self, source: str, sample: dict[str, Any]) -> dict[str, Any]:
        """원본 샘플을 messages/source 공통 포맷으로 변환합니다."""

        if source == "beomi/KoAlpaca-v1.1a":
            instruction = str(sample.get("instruction", ""))
            output = str(sample.get("output", ""))
            return {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "source": source,
            }

        if source == "dbdu/ShareGPT-74k-ko":
            conversations = sample.get("conversations", [])
            messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                role = "user" if turn.get("from") == "human" else "assistant"
                content = str(turn.get("value", ""))
                messages.append({"role": role, "content": content})

            return {"messages": messages, "source": source}

        question = str(sample.get("input", ""))
        answer = str(sample.get("output", ""))
        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": source,
        }
