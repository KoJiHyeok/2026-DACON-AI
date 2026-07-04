"""추론 코드 원형 — submit/script.py 는 이 파일을 기반으로 패키징한다.

서버 실행 규약 (베이스라인 script.py에서 확인된 계약):
    입력  : ./data/test.jsonl, ./data/sample_submission.csv (서버 제공)
    모델  : ./model/ (submit.zip에 동봉)
    출력  : ./output/submission.csv (sample_submission과 같은 id 순서·컬럼)

제약: 오프라인(네트워크 호출 금지), 추론 ≤ 10분, T4/3vCPU/12GB RAM.

TODO: 모델 확정 후 구현. 피처는 src/features.py 와 동일 로직 유지.
"""

if __name__ == "__main__":
    raise NotImplementedError
