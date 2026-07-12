import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DATA=Path(r"C:\dev\2026-AI-DACON\data\train.jsonl")
MODEL=Path(r"C:\dev\2026-AI-DACON\submit\model")
def load(p=DATA,n=30):
 a=[]
 for l in p.open(encoding='utf-8-sig'):
  if l.strip():a.append(json.loads(l))
  if len(a)>=n:break
 return a
