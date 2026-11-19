import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from evi_hgt_phase98.runtime import main
if __name__=='__main__':
    try:
        result=main()
        if isinstance(result,dict) and result.get('status') not in (None,'PASS','COMPLETE'):raise SystemExit(1)
    except Exception as exc:
        import traceback,uuid
        base=Path(__file__).resolve().parents[2]/'outputs/phase9/phase98/part2_1'
        if 'formal' in sys.argv and '--manifest' in sys.argv:
            candidate=Path(sys.argv[sys.argv.index('--manifest')+1]).resolve()
            part3=Path(__file__).resolve().parents[2]/'outputs/phase9/phase98/part3'
            if candidate.is_relative_to(part3):base=candidate.parent
        if '--registration' in sys.argv:
            candidate=Path(sys.argv[sys.argv.index('--registration')+1]).resolve()
            expected=Path(__file__).resolve().parents[2]/'outputs/phase9/phase98/part2_3/TEST_ONLY'
            if candidate.parent==expected:base=expected.parent
            expected=expected.parent.parent/'part2_4/TEST_ONLY'
            if candidate.parent==expected:base=expected.parent
        log=base/'launcher_errors'/(uuid.uuid4().hex+'.log')
        log.parent.mkdir(parents=True,exist_ok=True);log.write_text(traceback.format_exc(),encoding='utf-8')
        print('运行入口已拒绝或停止：',str(exc),'完整日志：',log,flush=True)
        raise SystemExit(1)
