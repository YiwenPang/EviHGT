"""Offline three-update integration check using an isolated synthetic corpus."""
import json
from pathlib import Path
from uuid import uuid4

import torch

from .io import write_json
from .runner import run
from .settings import ROOT, Settings
from .synthetic import make_corpus


def run_smoke(*, data_root=None, output_root=None) -> Path:
    """Run FULL on 15 invented nodes without reading raw data or run settings."""
    namespace = uuid4().hex
    derived = Path(data_root) if data_root is not None else ROOT / "data/processed/smoke"
    outputs = Path(output_root) if output_root is not None else ROOT / "outputs/smoke"
    corpus_directory = derived.expanduser().resolve() / namespace
    runtime_directory = outputs.expanduser().resolve() / namespace
    corpus_directory.mkdir(parents=True, exist_ok=False)
    runtime_directory.mkdir(parents=True, exist_ok=False)
    print("[SYNTHETIC_TEST_ONLY] EviHGT tiny 流程验证：15 个合成节点，"
          "CPU/FP32，3 次真实参数更新。", flush=True)
    manifest = make_corpus(corpus_directory)
    definition = json.loads(manifest.read_text("utf8"))
    if (definition.get("purpose") != "SYNTHETIC_TEST_ONLY"
            or definition.get("schedule", {}).get("G") != 1
            or sum(entry["rows"] for entry in definition.get("fact_files", [])) != 10):
        raise ValueError("SYNTHETIC_TEST_ONLY 语料必须保持固定 tiny 规模和三次更新预算")
    settings = Settings(manifest=str(manifest), output_dir=str(runtime_directory),
                        device="cpu", precision="fp32", checkpoint_every=1)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = run(settings, variant="FULL")
    finally:
        torch.set_num_threads(previous_threads)
    completion_path = result / "completion.json"
    completion = json.loads(completion_path.read_text("utf8"))
    if (completion.get("scope") != "SYNTHETIC_TEST_ONLY"
            or completion.get("variant") != "FULL"
            or any(completion.get(key) != 3
                   for key in ("updates", "total_updates", "updates_this_invocation"))):
        raise RuntimeError("SYNTHETIC_TEST_ONLY 流程未完成预定的三次主模型更新")
    completion["purpose"] = "SYNTHETIC_TEST_ONLY"
    write_json(completion_path, completion)
    print(f"[SYNTHETIC_TEST_ONLY] tiny 流程验证完成，结果目录：{result}", flush=True)
    return result
