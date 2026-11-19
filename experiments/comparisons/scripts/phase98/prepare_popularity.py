"""Prepare fixed training-only drug-degree counts; no evaluation or model run."""
from pathlib import Path
import sys
import traceback
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from evi_hgt_phase98.popularity_preparation import prepare_all_popularity


if __name__ == '__main__':
    try:
        prepare_all_popularity()
    except Exception:
        target = Path(__file__).resolve().parents[2]/'outputs/phase9/phase98/part2/popularity/failures'
        target.mkdir(parents=True,exist_ok=True)
        log = target/(uuid.uuid4().hex+'.traceback.log')
        with log.open('x',encoding='utf-8') as handle:
            handle.write(traceback.format_exc())
        print(f'训练计数准备失败，完整日志：{log}',file=sys.stderr,flush=True)
        raise SystemExit(1)
