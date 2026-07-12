import importlib,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DATA=Path(os.environ.get("SPEED_DATA", r"C:\dev\2026-AI-DACON\data\train.jsonl"))
MODEL=Path(os.environ.get("SPEED_MODEL_DIR", r"C:\dev\2026-AI-DACON\submit\model"))
PYTHON=os.environ.get("SPEED_PYTHON", r"C:\dev\2026-AI-DACON\.venv\Scripts\python.exe")
def mod():
 p=str(ROOT/'submit');sys.path.insert(0,p) if p not in sys.path else None
 return importlib.import_module('script')
def setup(m,x=MODEL,device='cpu'):
 m.MODEL=str(x);m.LINEAR_PKL=str(x/'linear'/'model.pkl');m.STACKER_DIR=str(x/'stacker')
 if device not in ('cpu','cuda'): raise ValueError('SPEED_DEVICE must be cpu or cuda')
 if device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']=''
 if device=='cuda' and not m.torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
def load(p=DATA,n=30):
 a=[]
 for l in p.open(encoding='utf-8-sig'):
  if l.strip():a.append(json.loads(l))
  if len(a)>=n:break
 return a

def env_int(name, default):
 try: return max(1, int(os.environ.get(name, default)))
 except ValueError: raise ValueError(f'{name} must be an integer')
