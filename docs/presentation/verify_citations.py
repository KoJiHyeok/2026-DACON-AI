"""Deterministic citation spot-check for presentation source claims."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
# Each tuple is (check id, display citation, repository-relative source, exact needles).
CHECKS = [
    ("C01", "reports/eda_distribution.md", "context/reports/eda_distribution.md", ["70,000", "9,429", "mean 7.42"]),
    ("C02", "reports/eda_distribution.md", "context/reports/eda_distribution.md", ["edit_file | 11,171 | 15.96%", "web_search | 1,273 | 1.82%"]),
    ("C03", "reports/eda_distribution.md", "context/reports/eda_distribution.md", ["불균형 **8.8x**"]),
    ("C04", "reports/forensics_r1.md", "context/reports/forensics_r1.md", ["0.03%", "0.0157%", "min_rows≥12", "0%"]),
    ("C05", "reports/forensics_r1.md", "context/reports/forensics_r1.md", ["7.40%", "49.7%", "2,575/5,181"]),
    ("C06", "reports/forensics_r1.md", "context/reports/forensics_r1.md", ["sess_sim_*", "64,975행/8,330세션", "5,025행/1,099세션"]),
    ("C07", "reports/forensics_r2.md", "context/reports/forensics_r2.md", ["H1 eligible 191행", "독립 재구성 일치"]),
    ("C08", "D-010", "context/decisions.md", ["세션의 51%가 6턴 초과", "12턴이 30.7%", "8.5%뿐", "평균 +4.5턴"]),
    ("C09", "exp #34", "context/experiments.md", ["hist6대조 e5solo 0.70066", "hist12 e5solo 0.73617", "격리 델타 +0.02150"]),
    ("C10", "D-003", "context/decisions.md", ["CV는 세션 프리픽스 GroupKFold"]),
    ("C11", "audit", "context/reports/third_party_sol_model_audit_2026-07-10.md", ["9,969행/1,350세션", "respond_only +2.94%p"]),
    ("C12", "exp #43", "context/experiments.md", ["row **−0.00268**", "세션균등 −0.00096", "MC200 −0.00109±0.00547", "[−0.00751,+0.00176]"]),
    ("C13", "exp #35", "context/experiments.md", ["0.7623 (+0.0143)", "전이율 ≈ 67%"]),
    ("C14", "exp #51", "context/experiments.md", ["solo **0.74994**", "row **+0.00360**", "[−0.00078,+0.00809]", "0.7621"]),
    ("C15", "hist12 deploy review", "context/reports/verify_hist12_deploy_2026-07-10.md", ["22 passed, 0 failed", "finding 0건", "e5=12/mBERT=6"]),
    ("C16", "model audit components", "context/reports/third_party_sol_model_audit_2026-07-10.md", ["| Linear |", "| AAR stacker |", "| e5-base |", "| mBERT |", "| AU specialist |"]),
    ("C17", "model audit weights", "context/reports/third_party_sol_model_audit_2026-07-10.md", ["| 25% |", "| 30% |", "| 20% |", "AU 행에서 90%"]),
    ("C18", "ledger #1", "context/submissions.md", ["| 1 |", "**0.71884**"]),
    ("C19", "ledger #5", "context/submissions.md", ["| 5 |", "**0.73310**", "**+0.0142**"]),
    ("C20", "ledger #6", "context/submissions.md", ["| 6 |", "**0.7400**", "**+0.0069**"]),
    ("C21", "ledger #7", "context/submissions.md", ["| 7 |", "**0.7467**", "4분14초"]),
    ("C22", "ledger #11", "context/submissions.md", ["| 11 |", "**0.7623**", "67% 전이"]),
    ("C23", "exp #50", "context/experiments.md", ["row **−0.00164**", "누수 교집합 0", "P(Δ>0)=0.230"]),
    ("C24", "D-013", "context/decisions.md", ["챔피언 0.7623", "최고 점수 자동 선택"]),
    ("C25", "exp #52 model", "context/experiments.md", ["| 52 | 07.13 |", "Qwen2.5-0.5B 디코더 분류기", "하이브리드(linear+AAR+Qwen블록+soft-AU, mBERT 제외"]),
    ("C26", "exp #52 runtime", "context/experiments.md", ["Qwen 연산활성 360M/24L", "86M×2/12L", "길이정렬 배칭 1.7x + fast_aar 2.8x", "둘 다 출력등가"]),
    ("C27", "ledger #13", "context/submissions.md", ["| 13 | 07-13 12:34 |", "**시간초과 FAIL**", "평가 T4에서 추론 10분 초과, 채점 불가"]),
    ("C28", "ledger #14", "context/submissions.md", ["| 14 | 07-13 16:03 |", "**0.77089**", "89→79등", "**73.5% 전이**"]),
    ("C29", "daily 07-13 timeout", "context/daily/2026-07-13.md", ["13:0x 제출 #13 시간초과 FAIL", "파라미터 수 등가 ≠ 연산 등가", "디코더 24층 + hist12 장문 시퀀스"]),
    ("C30", "daily 07-13 recovery", "context/daily/2026-07-13.md", ["17:4x 채점 완료 LB 0.77089", "+0.00853, 89→79등", "holdout +0.0116의 73.5% 전이", "T4 30k 리허설 515s"]),
    ("C31", "daily 07-13 equivalence", "context/daily/2026-07-13.md", ["패딩 연산 1.70x 절감", "150행 argmax 100%", "5,000행 오차 0.0", "argmax 100%"]),
    ("C32", "T4 rehearsal timing", "docs/t4_rehearsal.md", ["실평가 30,000행", "29.7s (구경로 ~84s)", "471.5s (구경로 ~800s)", "515s (8.6분)"]),
    ("C33", "T4 rehearsal compute", "docs/t4_rehearsal.md", ["활성 360M/24층", "각 ~86M/12층", "정렬 배칭 이득 1.7x"]),
    ("C34", "Qwen solo rejection", "context/experiments.md", ["solo 단독 vs 챔피언 블렌드: 혼합(row +0.0033, 세션균등 음수) — 단독 배치 부결"]),
]


def main() -> int:
    docs = [ROOT / "docs/presentation/sources.md", ROOT / "docs/presentation/key_numbers.md"]
    citation_pattern = re.compile(
        r"\((?:exp #\d+|D-\d{3}|대장 #\d+|daily \d{2}-\d{2}|docs/[^)]+|reports/[^)]+)\)"
    )
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        if not citation_pattern.search(text):
            raise AssertionError(f"no citations found: {doc}")
        lines = text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("|---"):
                continue
            if stripped.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|---"):
                continue
            is_claim = (
                stripped.startswith(("- ", "1. ", "2. ", "3. ", "4. ", "5. ", "|"))
                or not stripped.startswith(("```", ">"))
            )
            if is_claim and not citation_pattern.search(stripped):
                raise AssertionError(f"uncited claim line: {doc.relative_to(ROOT)}:{index + 1}: {stripped}")

    combined = "\n".join(doc.read_text(encoding="utf-8") for doc in docs)
    experiments = (ROOT / "context/experiments.md").read_text(encoding="utf-8")
    decisions = (ROOT / "context/decisions.md").read_text(encoding="utf-8")
    submissions = (ROOT / "context/submissions.md").read_text(encoding="utf-8")
    for value in re.findall(r"\(exp #(\d+)\)", combined):
        assert f"| {int(value)} |" in experiments, f"unresolved experiment citation: {value}"
    for value in re.findall(r"\(D-(\d{3})\)", combined):
        assert f"## D-{value}" in decisions, f"unresolved decision citation: {value}"
    for value in re.findall(r"\(대장 #(\d+)\)", combined):
        assert f"| {int(value)} |" in submissions, f"unresolved ledger citation: {value}"
    for value in re.findall(r"\(reports/([A-Za-z0-9_-]+\.md)\)", combined):
        assert (ROOT / "context/reports" / value).is_file(), f"unresolved report citation: {value}"
    for value in re.findall(r"\(daily (\d{2}-\d{2})\)", combined):
        path = ROOT / "context/daily" / f"2026-{value}.md"
        assert path.is_file(), f"unresolved daily citation: {value}"
    for value in re.findall(r"\(docs/([A-Za-z0-9_/-]+\.md)\)", combined):
        assert (ROOT / "docs" / value).is_file(), f"unresolved docs citation: {value}"

    for check_id, _citation, relpath, needles in CHECKS:
        source = (ROOT / relpath).read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in source]
        if missing:
            raise AssertionError(f"anchor suite {check_id} failed: {missing!r}")

    failures: list[str] = []
    print(f"checks={len(CHECKS)}/{len(CHECKS)} mode=exhaustive")
    print("| check | citation | source | result |")
    print("|---|---|---|---|")
    for check_id, citation, relpath, needles in CHECKS:
        source = (ROOT / relpath).read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in source]
        result = "PASS" if not missing else "FAIL: " + repr(missing)
        print(f"| {check_id} | {citation} | `{relpath}` | {result} |")
        if missing:
            failures.append(f"{check_id}: {missing!r}")

    if failures:
        print("\n".join(failures))
        return 1
    print("mismatches=0 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
