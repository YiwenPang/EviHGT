# Python API

Run these examples from the repository root. Install `requirements-preprocess.txt` to prepare from raw files, or `requirements.txt` to use a complete [portable corpus](data.md). The CLI automatically prepares missing inputs before a fresh training run; a direct call to `run()` expects the corpus to exist.

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

settings = load_settings()
report_directory = run(settings, mode="validate")
```

After validation, start a new training run.

```python
run_directory = run(settings, variant="FULL", mode="train")
```

To score a saved checkpoint, use the matching configuration and set `checkpoint_path` to the path of a trusted local checkpoint.

```python
run(settings, variant="FULL", mode="evaluate", checkpoint=checkpoint_path)
```

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

The runner checks checkpoint identity, restores optimizer and RNG state, applies device and precision settings, and writes evaluation outputs.

## Stable objects

| Object | Relevant interface |
| --- | --- |
| `prepare_raw` | Raw configuration, final manifest path, output directory and offline flag; prepares R0F1 or checks the raw build identity before reusing a corpus |
| `Settings` | Explicit paths, device, precision, optimizer and reporting settings |
| `PortableCorpus` | `model_config`, `pretrained`, `assembler`, `plan`, `panels`, `training_seed`, `identity` |
| `TrainingPlan` | `G`, `total_steps` and `step(i)` for the fixed sampling plan |
| `QueryAssembler` | `assemble(query, domain=..., device=...)`; target masking and bounded graph/path construction |
| `BuiltEviHGT` | `model`, `forward`, `metadata`; named instance-local ablation behavior |

See [data.md](data.md) for input schemas, [model.md](model.md) for implemented computations, and [reproducibility.md](reproducibility.md) for verification and run records.
