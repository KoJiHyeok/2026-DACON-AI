"""src/features.py 불변식 테스트.

실행: python -m pytest tests/  (pytest 없으면: python tests/test_features.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import ACTION_CLASSES, session_id


def test_action_classes_match_spec():
    assert len(ACTION_CLASSES) == 14
    assert len(set(ACTION_CLASSES)) == 14
    # 데이터 명세(talkboard 416930)의 클래스명과 일치
    assert "respond_only" in ACTION_CLASSES
    assert "lint_or_typecheck" in ACTION_CLASSES


def test_session_id_strips_step_suffix():
    assert session_id("sess_sim_20260522_028750-step_02") == "sess_sim_20260522_028750"
    assert session_id("sess_sim_20260522_006284-step_01") == "sess_sim_20260522_006284"


def test_session_id_groups_same_session():
    a = session_id("sess_sim_20260522_014415-step_06")
    b = session_id("sess_sim_20260522_014415-step_01")
    assert a == b


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("전부 통과")
