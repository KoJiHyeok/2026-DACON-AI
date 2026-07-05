# -*- coding: utf-8 -*-
"""AU 전용 linear full-train (exp #23→#24 제출용 아티팩트).

train.jsonl의 sess_au 5,025행 전체로 char_wb(3-5) 단독 + LinearSVC(C=1.0, balanced) 학습
→ submit/model/au_linear/model.pkl 저장. 구성은 밤샘 task4 우승 변형(au_only_char_C1,
soft α=0.9와 짝 — 리그 0.73314). serialize는 submit/au_route.py 단일 소스를 import.

(#23 제출본은 word+char C=0.5 하드 라우팅 = LB 0.7331 — 그 아티팩트는 대장 #5 zip에 보존)

실행: .venv\\Scripts\\python.exe scripts/au/train_full_au.py  (서버 미러 sklearn 1.8.0 필수)
"""
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submit"))
import au_route  # noqa: E402  (serialize 단일 소스)


def main():
    samples = []
    with (ROOT / "data" / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if au_route.is_au(obj.get("id", "")):
                    samples.append(obj)
    with (ROOT / "data" / "train_labels.csv").open(encoding="utf-8") as f:
        label_map = {row["id"]: row["action"] for row in csv.DictReader(f)}
    y = np.array([label_map[str(s["id"])] for s in samples], dtype=object)
    print(f"AU rows: {len(samples)}, classes: {len(set(y))}")
    assert len(samples) == 5025, f"AU 행수 {len(samples)} != 5025 (데이터 확인)"
    assert len(set(y)) == 14

    texts = [au_route.serialize(s) for s in samples]
    # task4 우승: char-only C=1.0 (word 뷰 제거 — au_grid에서 league 최고)
    union = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                            max_features=120_000, sublinear_tf=True, strip_accents="unicode")
    x = union.fit_transform(texts)
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
    clf.fit(x, y)

    out = ROOT / "submit" / "model" / "au_linear"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"union": union, "clf": clf}, out / "model.pkl")
    size = (out / "model.pkl").stat().st_size / 1e6
    print(f"saved {out / 'model.pkl'} ({size:.1f}MB)")

    # sanity: train 예측 자기재현 (과적합 수치 — 판정용 아님, 파이프라인 확인용)
    preds = au_route.predict({"union": union, "clf": clf}, samples[:500])
    acc = float(np.mean([p == t for p, t in zip(preds, y[:500])]))
    print(f"sanity train-subset acc: {acc:.3f} (파이프라인 정상 확인용)")


if __name__ == "__main__":
    main()
