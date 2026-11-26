# Python API

Run these examples from the repository root. Install `requirements-preprocess.txt` to prepare from raw files, or `requirements.txt` to use a complete [portable corpus](data.md). The CLI `python main.py` automatically prepares missing inputs before a fresh training run; a direct call to `run()` expects the corpus to exist. Both routes use the same trainer, development evaluator, ranking and explanation output, and HTML report generator.

## Prepare from local raw files

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))

from evi_hgt.preprocessing.pipeline import prepare_raw
from evi_hgt.settings import load_settings

settings = load_settings()
prepared = prepare_raw(
    "configs/raw.yaml",
    manifest_path=settings.resolve(settings.manifest),
    output_dir=settings.resolve(settings.output_dir),
    offline=False,
)
```

`prepare_raw()` builds and validates the original R0F1 corpus and returns its manifest path, identity and validation status. It performs no optimizer updates. Reusing an existing final corpus requires available raw files and matching source, recipe and build identities. To validate a corpus without raw sources, call `run(settings, mode="validate")`. Set `offline=True` to require complete pinned auxiliary caches. The raw YAML controls source paths, staging, caches and the feature-inference device; `settings.device` controls training. See [configuration and recovery](data.md#configure-paths-devices-and-downloads) before changing an existing preparation's inputs or environment.

## Validate or run through the shared workflow

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))

from evi_hgt.settings import load_settings
from evi_hgt.runner import run
from evi_hgt.reporting import load_report_settings

settings = load_settings()
report_settings = load_report_settings("configs/report.yaml")
report_directory = run(settings, mode="validate")
```

After validation, start a new training run.

```python
run_directory = run(settings, variant="FULL", mode="train", report_settings=report_settings)
```

To score a saved checkpoint, use the matching configuration and set `checkpoint_path` to the path of a trusted local checkpoint.

```python
run_directory = run(
    settings,
    variant="FULL",
    mode="evaluate",
    checkpoint=checkpoint_path,
    report_settings=report_settings,
)
```

Training and checkpoint evaluation return the new run directory after writing `report.html`, `summary.json`, the ranking CSV files and `explanations/index.json`. `completion.json` records `status: COMPLETE` after all report outputs succeed. Validation exits before creating a model or these prediction outputs.

You can also construct the report settings explicitly:

```python
from evi_hgt.reporting import ReportSettings

report_settings = ReportSettings(
    explanation_queries=10,
    paths_per_query=5,
    ranking_top_k=10,
)
```

These settings control the Top-K ranking extracts and explanation output. The full ranking CSV files retain all supplied candidates. Report settings do not alter training, candidate membership or the model's graph/path extraction budgets. Use `python main.py --smoke` for a three-update CPU integration check with synthetic inputs and no network or real databases.

## Construct the model for inspection

```python
from evi_hgt.data.corpus import load_corpus
from evi_hgt.model import build_model, VARIANTS

corpus = load_corpus("data/processed/corpus/manifest.json")
built = build_model(
    corpus.model_config,
    pretrained=corpus.pretrained,
    seed=corpus.training_seed,
    variant="FULL",
)
metadata = built.metadata()
```

The factory initializes an untrained PyTorch module at `built.model`. `built.forward(model_batch, path_batch=...)` returns predictions from both heads. Pass an explicit empty path batch when the query yields no legal path.

The runner checks checkpoint identity, restores optimizer and RNG state, applies device and precision settings, and writes evaluation and report outputs. An evaluation or a resume with no remaining updates records the supplied checkpoint path; it does not copy the checkpoint into the new run directory.

## Stable objects

| Object | Relevant interface |
| --- | --- |
| `prepare_raw` | Raw configuration, final manifest path, output directory and offline flag; prepares R0F1 or checks the raw build identity before reusing a corpus |
| `Settings` | Explicit paths, device, precision, optimizer and console settings |
| `ReportSettings`, `load_report_settings` | Explanation-query count, paths per query and ranking Top-K output limits; optional report YAML loader |
| `PortableCorpus` | `model_config`, `pretrained`, `assembler`, `plan`, `panels`, `training_seed`, `identity` |
| `TrainingPlan` | `G`, `total_steps` and `step(i)` for the fixed sampling plan |
| `QueryAssembler` | `assemble(query, domain=..., device=...)`; target masking and bounded graph/path construction |
| `BuiltEviHGT` | `model`, `forward`, `metadata`; named instance-local ablation behavior |

See [data.md](data.md) for input schemas, [model.md](model.md) for implemented computations, and [reproducibility.md](reproducibility.md) for verification and run records.
