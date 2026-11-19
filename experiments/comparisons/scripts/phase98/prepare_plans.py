"""Freeze metadata-only training plans; this script never imports a runner."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from evi_hgt_phase98.sampling import freeze_all

if __name__ == '__main__':
    result=freeze_all()
    print('15 cell计划封存完成；正式训练与评估授权仍为false。',flush=True)
