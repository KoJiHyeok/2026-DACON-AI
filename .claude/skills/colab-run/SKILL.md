---
name: colab-run
description: DACON 236694 Colab 학습/추론 작업 안내 — 검증된 전체 셀 시퀀스(pip → 데이터 준비 → holdout npz → env → 학습 → 다운로드)를 항상 완결된 형태로 출력한다. 사용자가 Colab 작업을 요청하거나 '/colab-run'을 치면 사용.
---

# Colab 실행 안내 (DACON 236694)

## 철칙

1. **항상 전체 셀을 출력한다** — "아까 드린 셀", "이 줄만 바꾸세요" 금지. 사용자는 여러 노트북·계정을 오가므로 모든 셀을 복사-실행 가능한 완결 형태로 제공한다.
2. 셀 순서를 번호로 명시한다: 셀 1 pip → 셀 2 데이터 준비 → 셀 3 holdout npz(holdout85 모드일 때만) → 셀 4 env → 셀 5 학습(`colab/mdeberta_finetune.py` 통짜) → (완료 후) 다운로드 셀.
3. env 셀에는 반드시 소실 사고 방지 assert 4종을 포함한다: Drive 마운트 / OUT이 Drive 경로 / `./data/train.jsonl` 존재 / GPU 런타임. holdout85 모드면 `./holdout_base.npz` assert 추가.
4. colab 스크립트 규약: required argparse 금지, env 폴백(`MDEB_*`), `MDEB_RESUME=1` 기본 (세션 끊김 재개).
5. 멀티 계정: 공유 폴더는 보조 계정 마운트에 안 보임 — Drive 웹에서 '바로가기 추가' 필요 (colab/README_colab.md §1.5).
6. 완료 신호: `[npz] ... final macro-F1=...` + `[DONE]`. **F1 값을 먼저 보고받고** npz/fp16 다운로드를 안내한다.

## 표준 셀 (여기서 복사해 그대로 출력)

### 셀 1 — pip
```python
!pip install -q "transformers>=4.51" accelerate sentencepiece
```

### 셀 2 — 데이터 준비 (Drive 깊이 4 탐색 + open.zip 폴백)
```python
# ===== 데이터 준비 셀 — Drive에서 train.jsonl/train_labels.csv 찾아 ./data에 배치 =====
import os, glob, shutil, zipfile
from google.colab import drive

if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')

os.makedirs('./data', exist_ok=True)
need = ['train.jsonl', 'train_labels.csv']

def find_in_drive(fname, max_depth=4):
    base = '/content/drive/MyDrive'
    for depth in range(1, max_depth + 1):
        for p in glob.glob(base + '/*' * depth):
            if os.path.basename(p) == fname and os.path.isfile(p):
                return p
    return None

for fname in need:
    dst = f'./data/{fname}'
    if os.path.exists(dst):
        print(f'[skip] {dst} 이미 있음'); continue
    src = find_in_drive(fname)
    if src:
        print(f'[copy] {src} -> {dst}')
        shutil.copy(src, dst)

missing = [f for f in need if not os.path.exists(f'./data/{f}')]
if missing:
    z = find_in_drive('open.zip')
    assert z, f'{missing} 못 찾음 + open.zip도 없음 — Drive에 데이터를 올려주세요'
    print(f'[unzip] {z}')
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            b = os.path.basename(m)
            if b in missing:
                with zf.open(m) as fin, open(f'./data/{b}', 'wb') as fout:
                    shutil.copyfileobj(fin, fout)
                print(f'  -> ./data/{b}')

for f in need:
    sz = os.path.getsize(f'./data/{f}') / 1e6
    print(f'[OK] ./data/{f} ({sz:.1f}MB)')
```

### 셀 3 — holdout_base.npz 배치 (holdout85 모드 전용, 셀 2 이후 실행)
```python
import os, shutil
if not os.path.exists('./holdout_base.npz'):
    src = find_in_drive('holdout_base.npz')
    assert src, 'Drive에 holdout_base.npz 없음'
    shutil.copy(src, './holdout_base.npz')
print('[OK] holdout_base.npz', round(os.path.getsize('./holdout_base.npz')/1e6, 1), 'MB')
```

### 셀 4 — env 템플릿 (모델·OUT·MODE만 작업에 맞게 채워서 전체 출력)
```python
import os, torch

os.environ['MDEB_MODEL'] = '<HF 모델 ID>'
os.environ['MDEB_OUT'] = '/content/drive/MyDrive/<작업별 새 폴더>'   # 기존 OUT 덮어쓰기 금지
os.environ['MDEB_MODE'] = 'holdout85'   # 리그 판정용 | 'full' = 제출용 70k 전량
os.environ['MDEB_EPOCHS'] = '2'
os.environ['MDEB_BATCH'] = '8'
os.environ['MDEB_ACCUM'] = '2'          # 유효 배치 16 고정
os.environ['MDEB_MAXLEN'] = '384'
os.environ['MDEB_LR'] = '2e-5'
os.environ['MDEB_RESUME'] = '1'
os.environ['MDEB_CKPT_STEPS'] = '2000'
os.environ['MDEB_GRAD_CKPT'] = '0'      # OOM 시 '1' + BATCH '4' ACCUM '4'

assert os.path.isdir('/content/drive/MyDrive'), 'Drive 마운트 안 됨'
assert os.environ['MDEB_OUT'].startswith('/content/drive/'), 'OUT이 Drive가 아님!'
assert os.path.exists('./data/train.jsonl'), '데이터 준비 셀 먼저!'
assert os.path.exists('./holdout_base.npz'), 'npz 배치 셀 먼저!'   # full 모드면 이 줄 제거
assert torch.cuda.is_available(), 'GPU 런타임 아님! 런타임 유형 변경 → T4'
print('OUT :', os.environ['MDEB_OUT'], '| MODEL:', os.environ['MDEB_MODEL'])
```

### 셀 5 — 학습
`colab/mdeberta_finetune.py` 파일 내용 **전체**를 붙여넣으라고 안내 (파일 경로만 알려주지 말고, 요청받으면 내용 전체를 출력).

### 다운로드 셀 — npz (holdout85 완료 후)
```python
import os
import numpy as np
from google.colab import drive, files

if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')

OUT = '/content/drive/MyDrive/<작업 폴더>'
NPZ = OUT + '/holdout_mdeb.npz'   # 스크립트 고정 파일명 (모델이 무엇이든 이 이름)
assert os.path.exists(NPZ), 'npz 없음 — 학습 로그 tail 확인'

from sklearn.metrics import f1_score
d = np.load(NPZ, allow_pickle=True)
acts = [str(a) for a in d['actions']]
pred = np.array(acts)[d['probs'].argmax(axis=1)]
f1 = f1_score([str(y) for y in d['y_true']], pred, average='macro')
print(f'[단독 macro-F1] {f1:.5f} (rows={len(d["ids"])})')
files.download(NPZ)
```

### 다운로드 셀 — fp16 모델 (full 모드 완료 후)
```python
import shutil
from google.colab import files
shutil.make_archive('/content/<이름>_fp16', 'zip', '/content/drive/MyDrive/<작업 폴더>/model_fp16')
files.download('/content/<이름>_fp16.zip')
```

## 참고 수치 (판정 기준)

- 홀드아웃 9,969행. 단독 F1 비교 기준: e5 프록시 0.70509 / mBERT 2ep 0.67147 / mdeberta 0.66998
- 리그 기준선(4-way+soft-AU): 0.73877 — 승격 게이트 +0.005
- 소요: base급 60k 2ep ≈ 2시간(T4 fp32), full 70k 2ep ≈ 2.5시간
