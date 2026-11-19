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
    raise SystemExit(cli_main(project_root=PROJECT_ROOT))


if __name__ == "__main__":
    main()
