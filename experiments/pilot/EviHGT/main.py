from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evi_hgt.cli import main as cli_main


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == '--competition':
        if len(sys.argv) != 3 or sys.argv[2] not in ('prepare', 'verify', 'run', 'status'):
            raise SystemExit('Competition accepts prepare/verify/run/status; frozen configuration only')
        from evi_hgt.competition.runner import main as competition_main
        raise SystemExit(competition_main(sys.argv[2]))
    if len(sys.argv) >= 2 and sys.argv[1] == '--phase85-part3a-r4':
        if len(sys.argv) != 3 or sys.argv[2] not in ('verify', 'run', 'verify-complete'):
            raise SystemExit('R4 accepts verify/run/verify-complete; no scientific overrides')
        from evi_hgt.phase85_part3a_r4.pilot import main as r4_main
        raise SystemExit(r4_main(sys.argv[2]))
    if len(sys.argv) >= 2 and sys.argv[1] == '--phase85-part3a-r3':
        if len(sys.argv) != 3 or sys.argv[2] not in ('verify', 'run', 'verify-complete'):
            raise SystemExit('R3 accepts verify, run, verify-complete; no scientific overrides')
        from evi_hgt.phase85_part3a_r3.pilot import main as r3_main
        raise SystemExit(r3_main(sys.argv[2]))
    if len(sys.argv) >= 2 and sys.argv[1] == '--phase85-part3a-r2':
        if len(sys.argv) != 3 or sys.argv[2] not in ('verify', 'run', 'verify-complete'):
            raise SystemExit('R2 accepts exactly verify, run, or verify-complete; no scientific overrides')
        from evi_hgt.phase85_part3a_r2.pilot import main as r2_main
        raise SystemExit(r2_main(sys.argv[2]))
    if len(sys.argv) >= 2 and sys.argv[1] == '--phase85-part3a-r1':
        if len(sys.argv) != 3 or sys.argv[2] not in ('verify', 'run', 'verify-complete'):
            raise SystemExit('R1 accepts exactly verify, run, or verify-complete; no scientific overrides')
        from evi_hgt.phase85_part3a_r1.pilot import main as r1_main
        raise SystemExit(r1_main(sys.argv[2]))
    raise SystemExit(cli_main(project_root=PROJECT_ROOT))


if __name__ == "__main__":
    main()
