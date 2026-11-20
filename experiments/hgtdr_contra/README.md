# HGTDR-adapted-contra

This adapter trains the HGTDR single-output model on recorded contraindications. Each R0 fold uses its original masked graph, training contraindication positives, and sampled unlabeled pairs. Evaluation uses the common inner development candidates. The loss encodes P and sampled U as 1 and 0. U retains its unlabeled status.

The training recipe uses AdamW with an initial learning rate of 0.003 and cosine annealing over 300 epochs. Each batch contains equal numbers of P and U. We select the checkpoint by disease-macro ROC-AUC, then AP, then the earlier epoch. The R0 summary averages each disease over its eligible folds and then weights diseases equally.

Across 314 evaluable diseases in the five inner development panels, HGTDR-adapted-contra reached ROC-AUC 0.964060, AP 0.942237 and NDCG@10 0.959349. This is the contraindication row of the [main development comparison](../../README.md#development-results).

## Training

The entry point uses the audited EviHGT research workspace with its ranking and sampling modules, the original R0 fold manifests, and the hash-checked native graph caches and features. It also requires the local HGTDR reconstruction used by that workspace and its CUDA environment. The HGTDR architecture is described in the [original paper](https://doi.org/10.1093/bioinformatics/btae349).

From this repository, use the interpreter configured for the research workspace.

```sh
python -m experiments.hgtdr_contra.run run --research-root /path/to/research-workspace
```

Use `--output` to choose a separate output directory under the research workspace's `outputs` directory. The runner saves training state at completed epochs and records the selected weights and contra panel metrics for each fold. Each checkpoint records the adapter source hashes and the identities of its fold inputs. Use this checkout to resume a run started with this adapter. The November 2026 run used the research checkout's `src/evi_hgt_hgtdr_contra/` package. Its `protocol.json` records the SHA-256 of all five original adapter files. Keep source and derived data, pair-level predictions, logs, and weights in the local workspace.

## Tests

The public CPU tests use synthetic candidates and checkpoints to check frozen candidate order, scoring, checkpoint selection and recovery, and disease-first aggregation. Run them from the repository root.

```sh
python -m pytest -q tests/test_hgtdr_contra.py experiments/hgtdr_contra/tests/test_run.py
```

The nine additional tests of data binding and native training require the research workspace's Python modules. Supply its source path to run them.

```sh
python -m pytest -q -o pythonpath=/path/to/research-workspace/src experiments/hgtdr_contra/tests/test_data.py experiments/hgtdr_contra/tests/test_trainer.py
```
