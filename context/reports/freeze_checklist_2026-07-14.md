# Freeze checklist — 2026-07-14

- 종합 판정: **FAIL**
- 집계: PASS 7 / WARN 1 / FAIL 1
- 종료 규칙: FAIL이 하나라도 있으면 exit 1; WARN만 있으면 exit 0

| 항목 | 상태 | 상세 |
|---|---|---|
| submit/ SHA256 매니페스트 | **PASS** | 25개 파일, 1931.6 MiB → C:\dev\2026-AI-DACON\context\reports\freeze_manifest_2026-07-14.json |
| public LB 최고 제출 | **PASS** | numeric LB 최고 = #18 (0.77301); 기대값 = #18 (0.77301) |
| mBERT rollback 자산 | **PASS** | 존재: C:\dev\2026-AI-DACON\colab_out\mbert_encoder2_backup |
| fast_aar rollback 자산 | **PASS** | 존재: C:\dev\2026-AI-DACON\submit\fast_aar.py |
| encoder serialize 계약 | **PASS** | C:\dev\2026-AI-DACON\submit\model\encoder\serialize_config.json: {"max_hist": 12}; 기대값 = {"max_hist": 12} |
| 서버 rollback 경로 | **PASS** | 문서 기재만(접속 시도 안 함): ~/models/champ_encoder_s42, ~/out/qwen05i_2ep_full |
| TEMP 제출 잔해 | **WARN** | 1개 발견(삭제하지 않음): C:\Users\wlgur\AppData\Local\Temp\dacon_submit_api-0.1.2-py3-none-any.whl (file, 4958 bytes) |
| 디스크 여유 | **PASS** | C:\: 11.33 GiB free (WARN 기준 10.00 GiB 미만) |
| git 작업트리 | **FAIL** | 미커밋 변경:<br> M context/submissions.md<br> M scripts/freeze_check/check.py |

## 사람 확인 요약

FAIL 항목이 있습니다. 자산을 자동 수정하지 않았습니다. 최종 선택 전 사람이 판단하세요.
