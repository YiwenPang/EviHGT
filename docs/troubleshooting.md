# Troubleshooting

Run the commands below from the repository root using the same Python interpreter selected in your IDE. Choose the check for the inputs you have.

For downloaded raw databases:

```text
python check_data.py
```

This read-only check uses the Python standard library. It verifies the required filenames, OptimusKG release 2.0 file sizes and SHA-256 values, and the DrugBank XML/XSD root elements. See the [download and placement guide](data.md#download-and-place-the-source-databases) for the paths and current access requirements.

For a prepared corpus, install the runtime dependencies and run:

```text
python main.py --mode validate
```

Corpus validation uses CPU memory for the input index and exits before model creation, training or CUDA/BF16 checks. Add `--config /path/to/run.yaml` if your corpus uses a nondefault configuration.

A successful raw-file check does not create `data/processed/corpus/manifest.json`. The release imports normalized artifacts with `prepare_data.py` or exports a completed research corpus with `export_data.py`; its raw-to-training preprocessing entry point is not yet published. Use the [supported preparation routes](data.md#supported-preparation-routes) when the required prepared inputs are available. To check the software without those inputs, follow the [public CPU checks](../README.md#try-the-public-checks).

Runtime failures print an English explanation and the absolute traceback-log path. Inspect that file first. Normal runs write `outputs/RUN_ID/logs/errors.log` and `events.jsonl`; early configuration/import failures use `outputs/logs/startup_errors.log`. Preparation and export commands have their own log directories. Preserve the original inputs and the failed run's logs while investigating.

## Missing inputs or preparation errors

| Symptom | Meaning and next step |
| --- | --- |
| `Missing file` from `check_data.py` | Check the extracted filename and its location relative to the repository root. For files in another checkout, use `python check_data.py --root /path/to/checkout`. Follow the [raw-file layout](data.md#download-and-place-the-source-databases). |
| `Does not match OptimusKG release 2.0` | The file size or SHA-256 differs from the recorded aggregate input. Confirm the release and aggregate filename, and check for an incomplete download. Preserve the existing source file while investigating. |
| `Cannot read the expected XML header` | Confirm that the archive was extracted and that the inputs are the full DrugBank XML and XSD, rather than a CSV, SDF, archive or downloaded web page. |
| `Prepared corpus not found` or `Prepared corpus is missing` | Set the YAML `manifest` field to a complete prepared corpus. The default is `data/processed/corpus/manifest.json`. Raw-file checks can pass while this corpus is still absent; see the [supported preparation routes](data.md#supported-preparation-routes). |
| `Unsupported corpus schema` | The trainer expects `EVIHGT_PORTABLE_CORPUS_V1`. Use `prepare_data.py` with a complete source map of normalized inputs, or `export_data.py` with the required completed research workspace. Renaming a raw-data manifest does not convert its schema. |
| `Unbound corpus input path` or `Missing corpus input` | A referenced file is absent or missing from the hashed inventory. Restore the matching corpus as a unit, including its manifest. |
| `Output already exists` | Choose a new derived-data directory, such as `data/processed/corpus_v2`. Failed preparations can leave `.attempt_...` directories for inspection. |
| `No completed masked corpus exists` or a missing sampling seal | Supply a completed, masked research corpus and its sealed sampling plan. |
| A semantic-declaration, channel, target-mask or conditional-record error | Check affected rows against the source audit and the relevant schema in [data.md](data.md). Keep U labels and verified relation meanings intact while correcting the input. |

The [source-map template](../schemas/source.example.json) and [D1 metadata template](../schemas/d1_metadata.example.json) describe the required prepared artifacts. Replace their placeholders with verified paths, hashes and declarations. The templates do not generate entity mappings, labels, features or sampling plans.

## Python environment, CUDA and BF16

Confirm which interpreter is running the program:

```text
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
```

If an import is missing, install dependencies into that interpreter with `python -m pip install -r requirements.txt`. In PyCharm, verify the project's configured interpreter and the run configuration use the same environment. See the [installation instructions](../README.md#installation) for PyTorch setup.

The default settings require `device: cuda` and `precision: bf16`. If you get `CUDA is unavailable`, check the PyTorch installation, GPU and driver used by that interpreter. `Configured BF16 is unsupported` requires an explicit FP32 configuration or a compatible environment.

For an intentional CPU/FP32 run:

```text
python main.py --config configs/cpu.yaml
```

For CUDA/FP32, create a separate configuration based on `configs/default.yaml` and set `precision: fp32` while keeping `device: cuda`. Start a new run when changing device or precision. Checkpoints bind to the device and precision used to create them.

RDKit is needed only when explicitly invoking the molecular-fingerprint helper. Supply the frozen D1 arrays with the prepared corpus.

## Paths, relocation and file hashes

Paths inside the corpus manifest must be relative POSIX paths within the corpus directory. Absolute paths, parent traversal, backslashes and escaping symlinks are rejected. Move the **complete** portable corpus together with its manifest; its content identity does not depend on the containing directory.

`manifest` and `output_dir` inside a YAML configuration resolve relative to the repository root. Explicit command-line paths such as `--config` and `--checkpoint` resolve from the process working directory, so use a correct working directory or an absolute path. The no-argument entry point locates its default configuration from the repository itself.

Old research manifests may contain their former absolute workspace root. Only the explicit migration utility handles this mapping:

```text
python export_data.py --research-root /new/research --recorded-root /old/research --cell R0F1 --d1-root /local/frozen/features --output data/processed/corpus
```

Use the actual former root recorded in those manifests, including its original Windows path form when applicable. This mapping is not needed by ordinary portable-corpus training.

A hash mismatch means a file differs from the recorded copy. Check for a partial copy, stale features, edits or files from another cell. Restore the matching file to resume a checkpoint, or audit and prepare a new corpus for a new run.

## Checkpoints and interrupted runs

Resume with a trusted local checkpoint produced by this release:

```text
python main.py --checkpoint outputs/RUN_ID/checkpoints/latest.pt
```

To score development panels without additional optimizer updates:

```text
python main.py --mode evaluate --checkpoint outputs/RUN_ID/checkpoints/latest.pt
```

Use the appropriate `--config` argument if the run used a nondefault configuration. Evaluation requires a checkpoint. For an ablation checkpoint, select exactly its variant:

```text
python ablation.py --variants NO_PATH --checkpoint outputs/RUN_ID/checkpoints/latest.pt
```

`Checkpoint model, inputs or training settings differ` is a binding failure. Compare the original run's `configuration.json`, `model.json` and corpus identity with the current invocation. In particular, variant, model/catalog configuration, seed, optimizer settings, device and precision must agree. You can change output location, display refresh, checkpoint interval and evaluation cadence when resuming.

The saved checkpoint includes model parameters, AdamW state, the logical update cursor and Python/NumPy/PyTorch RNG states. Resume starts at the next update after that saved cursor. Work after the most recent saved checkpoint may be repeated. Each invocation writes a new run directory, so look there for subsequent checkpoints and results. For a checkpoint at the end of the plan, use `--mode evaluate` to score the saved model.

Resume from a committed `latest.pt`. Leave any `latest.pt.partial` from an interrupted write out of recovery. If scoring was interrupted after saving a checkpoint, rerun evaluation with that checkpoint. Load only trusted checkpoints because deserialization can execute code.

## Memory and slow initialization

Input loading hashes all declared files, builds a CPU evidence index and validates features and the complete fixed schedule. Large files can take substantial time before a file-count bar advances. Named stages without a known total display elapsed time rather than a percentage in a terminal. GPU training starts after these input checks.

The input index implements accounting limits of 8 GiB for retained static payloads and 2 GiB for build scratch. Total process memory can exceed those builder limits. A `static immutable input hard budget exceeded` or `startup scratch hard budget exceeded` error requires inspecting corpus size and schema, especially unexpectedly expanded property records. Confirm that the input is the intended masked corpus and preserve its evidence records when diagnosing the failure.

For CUDA out-of-memory failures, check other GPU workloads and inspect the failing group's graph/query counts. The runner assembles the original positive group and its declared extra U queries before backward, so group size also affects memory. The two-hop, 1000-node and 5000-message bounds apply per query. The console reports GPU memory allocated by PyTorch.

Keep the original group, graph bounds, precision and plan when attempting to reproduce a run. Arbitrarily reducing graphs, splitting groups or dropping paths changes the scientific computation. An explicit CPU/FP32 run is available for a new portability run, with different runtime and numerical behavior. See [the model card](model_card.md#known-limits-and-reproducibility).

## Path and numerical errors

An empty path set is valid. The model receives an explicit empty packed path batch and uses the exact graph-only fallback. If a path batch is missing, or a direction, forbidden fact or target mapping fails validation, check graph assembly and the prepared target map.

Nonfinite predictions, losses or gradients fail the run and retain the traceback. Check the recorded environment, finite feature values, availability masks and the preceding structured update events. The console displays very small finite losses in scientific notation.

## Metrics that are unavailable or look unexpectedly high

`N/A` is intentional when a metric has no evaluable disease. Inspect `metrics.json` for `NO_P`, `NO_U`, skipped disease details and candidate counts. Missing or nonfinite scores and duplicate pairs fail validation.

AP and trapezoid AUPR use different integration definitions and can differ when scores are tied. `Hits@5` or `Hits@10` can be guaranteed in a small candidate panel. Review `effective_k`, P/U counts and the recorded random-reference fields before attributing a high value to model quality. See [how to interpret development metrics](model_card.md#evaluation-interpretation).

## Terminal, PyCharm and redirected output

The display updates in place when the output stream is a terminal and keeps completed lines. In PyCharm, enable **Emulate terminal in output console** when available if an updating bar is desired. A redirected stream or ordinary non-TTY IDE console receives plain stage messages and periodic summaries for counted work, without terminal control codes. Unknown-duration stages may show only their start and completion in redirected output.

If redirected output pauses, check the active stage and `logs/events.jsonl`. The runner logs optimizer events after completed updates.

When reporting a reproducible software issue, include the command, exception/log excerpt, package/environment versions, variant and non-sensitive configuration or schema details. Remove local identifiers and restricted content from the excerpt. Use a small synthetic fixture in place of licensed inputs or real checkpoints.
