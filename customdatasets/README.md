# customdatasets

이 폴더는 프로젝트 실험에서 공통으로 사용하던 3개 한국어 데이터셋을 하나의 허깅페이스 데이터셋으로 결합하기 위한 로컬 빌더를 제공합니다.

## 포함 데이터셋

- `beomi/KoAlpaca-v1.1a`
- `dbdu/ShareGPT-74k-ko`
- `kikikara/ko_QA_dataset`

## 로드 예시

```python
from datasets import load_dataset

dataset = load_dataset("customdatasets")
train_split = dataset["train"]
print(train_split[0])
```

통합 스키마는 아래 2개 컬럼을 사용합니다.

- `messages`: `[{"role": str, "content": str}, ...]`
- `source`: 원본 데이터셋 이름
