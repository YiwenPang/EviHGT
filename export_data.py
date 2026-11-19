"""Export one completed research cell to the standalone portable input contract."""
from pathlib import Path
import argparse
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root',type=Path,required=True,help='Explicit original research workspace with completed derived artifacts')
    parser.add_argument('--cell',required=True,help='Original cell ID, for example R0F1')
    parser.add_argument('--d1-root',type=Path,required=True,help='Frozen D1 features directory containing manifest.json and typed arrays')
    parser.add_argument('--recorded-root',help='Former workspace root stored in old manifests, only needed after relocation')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/corpus',help='New derived corpus directory; must not exist')
    args = parser.parse_args(argv)
    try:
        from evi_hgt.data.research_export import export_research
        from evi_hgt.console import Console
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        with Console(ROOT/'outputs/logs'/('export_'+stamp)) as console:
            console.banner()
            try:
                with console.phase(1,'Export prepared research cell'):
                    with console.stage_progress('Exporting') as progress:
                        result = export_research(args.research_root,args.cell,args.d1_root,args.output,
                            recorded_root=args.recorded_root,progress=progress)
                console.say('Portable manifest: '+result['manifest'])
                return 0
            except Exception as exc:
                console.failure(exc)
                return 1
    except Exception as exc:
        log_dir = ROOT/'outputs/logs'
        log_dir.mkdir(parents=True,exist_ok=True)
        path = log_dir/('export_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.log')
        path.write_text(traceback.format_exc(),encoding='utf-8')
        print('Input export failed: '+str(exc)+'\nComplete log: '+str(path),file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
