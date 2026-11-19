"""CPU read-only payload/source checks. Does not construct or execute any model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from evi_hgt_phase98.reuse import verify_all_reuse
from evi_hgt_phase98.native import seal_native_contracts

if __name__=='__main__':
    verify_all_reuse()
    seal_native_contracts()
