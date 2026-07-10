# -*- coding: utf-8 -*-
"""EDA 분포 분석 (재현용) — DACON 236694, AI agent next-action 14-class.

읽기 전용. `data/train.jsonl` + `data/train_labels.csv` 필요.
결과 요약을 stdout으로 출력. 상세 해석은 context/reports/eda_distribution.md 참고.

실행: python notebooks/eda_distribution.py
"""
import json, re, sys, io
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA = Path(__file__).resolve().parents[1] / "data"
HANGUL = re.compile(r"[가-힣]")


def sess_id(i):
    return i.rsplit("-step_", 1)[0]


def step_no(i):
    m = re.search(r"-step_(\d+)$", i)
    return int(m.group(1)) if m else -1


def ko_ratio(s):
    if not s:
        return 0.0
    letters = [c for c in s if not c.isspace()]
    if not letters:
        return 0.0
    return sum(bool(HANGUL.match(c)) for c in s) / len(letters)


def result_flag(res):
    low = (res or "").lower()
    if "fail" in low or "error" in low or " err" in low:
        return "fail"
    if "pass" in low or low.startswith("ok") or "ok;" in low or "success" in low:
        return "ok"
    return "other"


def load():
    lab = dict(pd.read_csv(DATA / "train_labels.csv").values)
    rows = []
    with open(DATA / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sm = d.get("session_meta", {}) or {}
            ws = sm.get("workspace", {}) or {}
            hist = d.get("history", []) or []
            acts = [h.get("name") for h in hist if h.get("role") == "assistant_action"]
            last_res = next((h.get("result_summary") or "" for h in reversed(hist)
                             if h.get("role") == "assistant_action"), "")
            lm = ws.get("language_mix", {}) or {}
            cp = d.get("current_prompt", "") or ""
            rows.append(dict(
                id=d["id"], sid=sess_id(d["id"]), step=step_no(d["id"]),
                action=lab.get(d["id"]), n_hist=len(hist),
                last_act=acts[-1] if acts else "<none>",
                prev_act=acts[-2] if len(acts) >= 2 else "<none>",
                res_flag=result_flag(last_res),
                turn_index=sm.get("turn_index"), budget=sm.get("budget_tokens_remaining"),
                user_tier=sm.get("user_tier"), lang_pref=sm.get("language_pref"),
                git_dirty=ws.get("git_dirty"), ci=ws.get("last_ci_status"),
                n_open=len(ws.get("open_files", []) or []),
                top_lang=max(lm, key=lm.get) if lm else "<none>",
                cp=cp, cp_len=len(cp), cp_ko=ko_ratio(cp),
            ))
    return pd.DataFrame(rows)


def cramers_v(a, b):
    ct = pd.crosstab(a, b).values.astype(float)
    n = ct.sum()
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = ((ct - exp) ** 2 / exp).sum()
    r, k = ct.shape
    return np.sqrt((chi2 / n) / min(r - 1, k - 1))


def main():
    df = load()
    N = len(df)
    print(f"rows={N:,}  sessions={df.sid.nunique():,}  missing_label={df.action.isna().sum()}")

    print("\n[클래스 분포]")
    print((df.action.value_counts(normalize=True) * 100).round(2).to_string())

    print("\n[첫 턴 vs 전체 lift]")
    fv = df[df.n_hist == 0].action.value_counts(normalize=True)
    av = df.action.value_counts(normalize=True)
    print((fv / av).sort_values(ascending=False).head(6).round(2).to_string())

    print("\n[전이: last_act → 정답 top1]")
    ctn = pd.crosstab(df.last_act, df.action, normalize="index")
    for la in df.last_act.value_counts().index:
        r = ctn.loc[la].sort_values(ascending=False)
        print(f"  {la:>16} -> {r.index[0]} {r.iloc[0]*100:.0f}%")

    print("\n[result_flag 분기 (run_tests / lint)]")
    for la in ["run_tests", "lint_or_typecheck"]:
        for rf in ["ok", "fail"]:
            s = df[(df.last_act == la) & (df.res_flag == rf)]
            if len(s) < 30:
                continue
            t = s.action.value_counts(normalize=True).head(2)
            print(f"  {la} [{rf}] -> " + ", ".join(f"{a} {p*100:.0f}%" for a, p in t.items()))

    print("\n[메타 연관 강도 Cramér's V]")
    df["bigram"] = df.prev_act + ">" + df.last_act
    df["ob"] = pd.cut(df.n_open, [-1, 0, 1, 3, 100], labels=["0", "1", "2-3", "4+"])
    df["tb"] = pd.cut(df.turn_index, [-1, 2, 5, 9, 100], labels=["1-2", "3-5", "6-9", "10+"])
    for col in ["ob", "bigram", "tb", "git_dirty", "last_act", "ci", "top_lang",
                "user_tier", "lang_pref"]:
        sub = df.dropna(subset=[col])
        print(f"  {col:>10}: {cramers_v(sub[col], sub.action):.3f}")

    print("\n[프롬프트 길이/한글비율 by action]")
    g = df.groupby("action").agg(mean_len=("cp_len", "mean"),
                                 ko=("cp_ko", lambda s: (s > 0.1).mean()))
    print(g.round(2).sort_values("mean_len").to_string())

    print("\n[Train/Test caveat]")
    test_ids = [json.loads(l)["id"] for l in open(DATA / "test.jsonl", encoding="utf-8")]
    overlap = sum(i in set(df.id) for i in test_ids)
    print(f"  local test.jsonl: {len(test_ids)}건, train과 중복 {overlap}/{len(test_ids)} "
          f"→ 스모크 샘플. 신뢰 지표는 세션 GroupKFold OOF 뿐.")


if __name__ == "__main__":
    main()
