# smollm2ko

한국어 데이터셋 준비와 가공을 위한 저장소입니다.

## 데이터셋 구조

`datasets` 폴더에서 원본, 중간 산출물, 최종 가공본, 테스트 프롬프트를 관리합니다.
자세한 규칙은 `datasets/README.md`를 참고하세요.

## raw 데이터셋 점검

raw 데이터셋 폴더의 경로, 총 크기, split, 테이블 구조, 샘플 1건을 확인합니다.

```bash
source .venv/bin/activate
python scripts/inspect_raw_datasets.py
```
