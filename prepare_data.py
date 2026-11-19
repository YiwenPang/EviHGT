"""Validate and package an explicit map of licensed, normalized local inputs."""
from pathlib import Path
import argparse
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='EVIHGT_SOURCE_MAP_V1 JSON; paths resolve relative to this file')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/corpus',help='New derived corpus directory; existing paths are never overwritten')
    args = parser.parse_args(argv)
    try:
        from evi_hgt.data.prepare import prepare_from_spec
        from evi_hgt.console import Console
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        with Console(ROOT/'outputs/logs'/('prepare_'+stamp)) as console:
            console.banner()
            try:
                with console.phase(1,'Prepare and validate normalized inputs'):
                    with console.stage_progress('Preparing') as progress:
                        result = prepare_from_spec(args.source,args.output,progress=progress)
                console.say('Portable manifest: '+result['manifest'])
                return 0
            except Exception as exc:
                console.failure(exc)
                return 1
    except Exception as exc:
        log_dir = ROOT/'outputs/logs'
        log_dir.mkdir(parents=True,exist_ok=True)
        path = log_dir/('prepare_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.log')
        path.write_text(traceback.format_exc(),encoding='utf-8')
        print('Input preparation failed: '+str(exc)+'\nComplete log: '+str(path),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
