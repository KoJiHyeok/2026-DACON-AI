"""학습 엔트리포인트.

data/train.jsonl + data/train_labels.csv 를 읽어 모델을 학습하고
submit/model/ 에 아티팩트를 저장한다.

- 피처는 src/features.py 만 사용 (추론과 단일 소스)
- CV: 세션 프리픽스 기준 GroupKFold, 지표는 Macro-F1
- 시드·버전 고정 (본선 코드 검증 대비 재현성 필수)

TODO: EDA 완료 후 Tier 0(베이스라인 재현)부터 구현.
"""

if __name__ == "__main__":
    raise NotImplementedError
