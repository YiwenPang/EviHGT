"""Launch TxGNN in its independent, pinned CPU runtime."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from evi_hgt.io import write_json
from evi_hgt.settings import ROOT


def _interpreter(context):
    default = ROOT / (".venv-txgnn/Scripts/python.exe" if os.name == "nt" else ".venv-txgnn/bin/python")
    interpreter = Path(context.get("txgnn_python") or os.environ.get("EVIHGT_TXGNN_PYTHON") or default)
    if not interpreter.is_file():
        raise FileNotFoundError(
            f"TxGNN CPU interpreter not found at {interpreter}. Install requirements-txgnn.txt "
            "in .venv-txgnn or set EVIHGT_TXGNN_PYTHON to a compatible CPython 3.11 interpreter."
        )
    return interpreter.resolve()


def _environment():
    environment = dict(os.environ, DGL_NUM_THREADS="1", OMP_NUM_THREADS="1", DGLBACKEND="pytorch")
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def _worker_context(context):
    result = {name: str(value) if isinstance(value, Path) else value
              for name, value in context.items() if name in {"native_cache", "txgnn_source", "offline"}}
    result["_txgnn_worker"] = True
    return result


def check_worker(context, *, smoke):
    arguments = [str(_interpreter(context)), "-m", "experiments.workflow.native_sources.txgnn", "--check"]
    payload = json.dumps(dict(context=_worker_context(context), smoke=smoke))
    subprocess.run(arguments, input=payload.encode("utf-8"), cwd=ROOT, env=_environment(), check=True)


def run_worker(settings, *, smoke, checkpoint, context):
    interpreter, environment = _interpreter(context), _environment()
    output = settings.resolve(settings.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".txgnn_request_", dir=output) as temporary:
        request = Path(temporary) / "request.json"
        response = Path(temporary) / "response.json"
        write_json(request, dict(settings=replace(settings, device="cpu", precision="fp32").as_dict(),
                                checkpoint=str(checkpoint) if checkpoint else None, smoke=smoke,
                                context=_worker_context(context), response=str(response)))
        subprocess.run([str(interpreter.resolve()), "-m", "experiments.workflow.native_sources.txgnn", str(request)],
                       cwd=ROOT, env=environment, check=True)
        result = Path(json.loads(response.read_text("utf-8"))["run"])
        if not result.resolve().is_relative_to(output) or not (result / "completion.json").is_file():
            raise ValueError("TxGNN worker returned an incomplete result")
        return result
