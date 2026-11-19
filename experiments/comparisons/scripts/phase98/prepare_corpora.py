"""CPU-only authorized Part2.1 cell preparation and real data-loader acceptance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from evi_hgt_phase98.corpus import prepare_all

if __name__=='__main__':
    print(prepare_all(),flush=True)
