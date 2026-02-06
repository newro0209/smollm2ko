# smollm2ko

한국어 데이터셋 준비와 가공을 위한 저장소입니다.

## 데이터셋 점검 및 통합 스키마 지시문

데이터셋을 선정한 뒤, AI에게 `scripts/inspect_hf_datasets.py`를 **지시하여 실행**하게 합니다.

이를 통해 각 데이터셋의 구조와 필드 특성을 체계적으로 파악하고, 그 결과를 바탕으로 **Lazy Transform + `set_transform` 방식**을 활용해 여러 데이터셋을 **하나의 통합 스키마로 정규화하는 코드**를 작성하도록 합니다.

이 접근 방식은 특히 **이미 존재하는 데이터만을 활용하는 경우**, 별도의 전처리 스크립트나 파생 코드, 잡한 폴더 구조를 만들 필요를 줄여 전체 파이프라인을 더 단순하고 일관되게 유지하는 데 도움이 됩니다.

### 사용하려는 데이터

- beomi/KoAlpaca-v1.1a
- dbdu/ShareGPT-74k-ko
- kikikara/ko_QA_dataset

## 🧠 Vocabulary Adaptation 용어 정리표

| 단계    | 용어 (영문)                | 한글 표현     | 권장 변수명 (PEP8)        | 의미                  | 사용 빈도 |
| ----- | ---------------------- | --------- | -------------------- | ------------------- | ----- |
| 분석    | **low-utility tokens** | 저효용 토큰    | `low_utility_tokens` | 코퍼스에서 거의 쓰이지 않는 토큰  | ⭐⭐⭐   |
| 후보 선정 | **prunable tokens**    | 제거 가능 토큰  | `prunable_tokens`    | 제거해도 성능 영향이 적은 토큰   | ⭐⭐⭐⭐⭐ |
| 교체 확정 | **evicted tokens**     | 퇴출 예정 토큰  | `evicted_tokens`     | vocab ID 자리를 비워줄 토큰 | ⭐⭐⭐⭐  |
| 제거 완료 | **removed tokens**     | 제거된 토큰    | `removed_tokens`     | vocab에서 실제 삭제된 상태   | ⭐⭐    |
| 유지 대상 | **kept tokens**        | 유지 토큰     | `kept_tokens`        | 기존 임베딩을 그대로 사용하는 토큰 | ⭐⭐⭐⭐  |
| 유입 후보 | **incoming tokens**    | 유입 토큰     | `incoming_tokens`    | 새 vocab에 들어올 토큰     | ⭐⭐⭐⭐⭐ |
| 삽입 준비 | **injected tokens**    | 삽입 예정 토큰  | `injected_tokens`    | ID에 매핑되기 직전 상태      | ⭐⭐⭐   |
| 교체 완료 | **replaced tokens**    | 교체된 토큰    | `replaced_tokens`    | 기존 ID에 새 의미가 들어간 상태 | ⭐⭐⭐⭐⭐ |
| 최종 결과 | **adapted vocabulary** | 적응된 어휘    | `adapted_vocab`      | 교체가 완료된 최종 vocab    | ⭐⭐⭐⭐  |
| 작업 개념 | **embedding surgery**  | 임베딩 공간 수술 | `embedding_surgery`  | ID 유지 + 의미 교체 작업    | ⭐⭐⭐⭐  |

---

### ✅ 가장 표준적으로 쓰이는 핵심 4개

| 핵심 용어               |
| ------------------- |
| **prunable tokens** |
| **evicted tokens**  |
| **incoming tokens** |
| **replaced tokens** |
