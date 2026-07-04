"""피처 추출 — 학습(train.py)과 추론(infer.py/script.py)이 공유하는 단일 소스.

학습·추론 간 전처리 불일치는 조용한 점수 하락의 주범이므로,
샘플(dict) → 모델 입력 변환은 반드시 이 모듈의 함수만 사용한다.

입력 샘플 스키마 (train.jsonl / test.jsonl 공통):
    id, current_prompt, history(0~12턴), session_meta
"""

ACTION_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]


def session_id(sample_id: str) -> str:
    """id에서 세션 프리픽스 추출 (GroupKFold 그룹 키).

    예: 'sess_sim_20260522_028750-step_02' -> 'sess_sim_20260522_028750'
    """
    return sample_id.rsplit("-step_", 1)[0]


def extract_features(sample: dict) -> dict:
    """샘플 하나 → 피처 dict. TODO: EDA 후 구현."""
    raise NotImplementedError
