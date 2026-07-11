"""DACON 코드 제출 업로더 — make_submit.py가 만든 zip을 공식 제출 API로 업로드한다.

토큰/팀명은 리포에 절대 넣지 않는다. Windows 사용자 환경변수에서 읽는다:
    setx DACON_TOKEN "<마이페이지-계정관리에서 발급한 토큰>"
    setx DACON_TEAM  "<대회 팀 탭의 팀명>"
(setx 이후 새로 띄운 셸에만 반영되므로, 현재 프로세스에 없으면 레지스트리에서 직접 읽는다)

의존성 (PyPI에 없음 — DACON 자체 배포 wheel):
    pip install https://cfiles.dacon.co.kr/competitions/api/dacon_submit_api-0.1.2-py3-none-any.whl

Usage:
    python scripts/dacon_submit.py --check                     # 토큰·팀·남은 횟수만 확인 (제출 소모 없음)
    python scripts/dacon_submit.py --file submit.zip --memo "..." --yes   # 실제 제출

--yes 없이 --file을 주면 validate까지만 하고 멈춘다 (실수 방지).
"""
import argparse
import os
import sys

import requests

CPT_ID = "236694"
VALIDATE_URL = "https://app.dacon.io/api/v1/code-submission/validate"


def read_credential(name):
    """환경변수 → (Windows) HKCU\\Environment 레지스트리 순으로 읽는다."""
    val = os.environ.get(name)
    if val:
        return val.strip()
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                val, _ = winreg.QueryValueEx(key, name)
                return val.strip()
        except OSError:
            pass
    return None


def load_credentials():
    token = read_credential("DACON_TOKEN")
    team = read_credential("DACON_TEAM")
    missing = [n for n, v in [("DACON_TOKEN", token), ("DACON_TEAM", team)] if not v]
    if missing:
        print(f"[FAIL] 자격 정보 없음: {', '.join(missing)}")
        print('  발급: dacon.io 마이페이지 → 계정관리 → 개인 Token')
        print('  설정: setx DACON_TOKEN "<토큰>"  /  setx DACON_TEAM "<팀명>"')
        sys.exit(1)
    return token, team


def validate(token, team):
    """제출을 소모하지 않는 사전 검증 — 남은 횟수(quota)·용량 한도를 반환."""
    try:
        resp = requests.post(
            VALIDATE_URL,
            data={"cptId": CPT_ID, "teamName": team, "apiToken": token},
            timeout=10,
        )
        body = resp.json()
    except requests.RequestException as e:
        print(f"[FAIL] validate 요청 실패 (네트워크): {e}")
        sys.exit(1)
    except ValueError:
        print(f"[FAIL] validate 응답이 JSON이 아님 (HTTP {resp.status_code}) — 서버 점검/차단 가능성")
        sys.exit(1)
    if resp.status_code == 400:
        print(f"[FAIL] validate 거부: {body.get('message', body)}")
        sys.exit(1)
    quota = body.get("quota", 0)
    limit = body.get("upload_filesize_limit")
    limit_gb = f"{limit / 1024**3:.2f}GB" if limit else "?"
    print(f"[OK] 팀 '{team}' / 대회 {CPT_ID} — 오늘 남은 제출 {quota}회, 용량 한도 {limit_gb}")
    if quota == 0:
        print("[FAIL] 오늘 제출 횟수 소진")
        sys.exit(1)
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate만 수행 (제출 소모 없음)")
    ap.add_argument("--file", help="제출할 zip 경로 (make_submit.py 산출물)")
    ap.add_argument("--memo", default="", help="제출 메모 — exp 번호·레시피 요약 권장")
    ap.add_argument("--yes", action="store_true", help="실제 업로드 실행 (없으면 validate까지만)")
    args = ap.parse_args()

    token, team = load_credentials()
    validate(token, team)

    if args.check or not args.file:
        return

    path = os.path.abspath(args.file)
    if not os.path.exists(path):
        print(f"[FAIL] 파일 없음: {path}")
        sys.exit(1)
    size_mb = os.path.getsize(path) / 1024**2
    print(f"[제출 대상] {path} ({size_mb:.0f}MB) memo={args.memo!r}")

    if not args.yes:
        print("[중단] --yes 없음 — validate까지만 수행했습니다. 실제 제출은 --yes를 붙이세요.")
        return

    from dacon_submit_api import dacon_submit_api

    result = dacon_submit_api.post_code_submission_file(path, token, CPT_ID, team, args.memo)
    print(result)
    if not result.get("isSubmitted"):
        sys.exit(1)
    print("[OK] 제출 완료 — context/submissions.md 대장에 기록하세요 (LB 점수는 채점 후 갱신).")


if __name__ == "__main__":
    main()
