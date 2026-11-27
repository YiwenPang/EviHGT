# Experiment code

[中文指南](README.zh-CN.md)

## Run from the source databases

Download the four source files and run the commands from the repository root. Preparation constructs the canonical graph, splits, candidate files, features, and training plans. The first online run downloads any missing pinned Open Targets and encoder resources. [Download and placement](../docs/data.md#download-and-place-the-source-databases) · [Main installation](../README.md#installation)

The `requirements-preprocess.txt` installation includes `filelock` for the experiment coordinator. Install `requirements-experiments.txt` for the matched controls and HGTDR. HGTDR also needs platform-matched `torch-sparse` and `torch-scatter`. TxGNN uses a separate CPython 3.11 environment and a pinned checkout. [Native installation and source cache](workflow/native_sources/README.md)

| Command | Default jobs |
| --- | --- |
| `python main.py --experiment ablations` | `FULL`, `CONTENT_ZERO`, and `FACT_QUALITY_CONSTANT` on `R0F1` and `R1F0`. [Evidence ablations](ablations/) |
| `python main.py --experiment losses` | FULL with `softplus` and `sigmoid` on `R0F0`, `R0F1`, and `R1F0`. The loss study uses the `DEV95` recipe for `R0F0`. [Loss comparison](ablations/) |
| `python main.py --experiment comparisons` | FULL and six baselines on `R0F0` through `R0F4`. [Baseline methods and IDs](comparisons/) |
| `python main.py --experiment cold` | Drug, disease, double-cold, and first-approval-year cases. The disease case also compares FULL with the HGTDR support and contraindication models on common test candidates. [Cold-start protocol](cold_start/) |
| `python main.py --experiment all` | The four suites in sequence. An ablation and loss arm with the same settings and fold runs once. |

Use `--cells` to select folds for an ablation, loss, or comparison run. Use `--variants` with `ablations`, `--methods` with `comparisons`, and `--cases` with `cold`. With `all`, each flag filters its corresponding suite. Pass IDs separated by spaces. Add `--skip-cold-hgtdr` to run the EviHGT cold cases without the HGTDR comparison.

```sh
python main.py --experiment ablations --cells R0F1 --variants FULL CONTENT_ZERO
python main.py --experiment losses --cells R0F0
python main.py --experiment cold --cases cold_disease --run-dir outputs/cold-disease
```

Use `--mode prepare` to build and verify the selected inputs. The default `train` mode continues through training, evaluation, and reporting. Derived corpora go under `data/processed/experiments/` and shared preparation stages under `data/interim/experiments/`. Select settings with `--config` and raw-data paths or feature preparation options with `--raw-config`. Use `--offline` when the pinned resources are cached. TxGNN accepts `--txgnn-python` for its interpreter and `--txgnn-source` for a verified local source checkout. [Raw configuration](../docs/data.md) · [TxGNN setup](workflow/native_sources/README.md)

A run defaults to `outputs/experiments/<timestamp>_<experiment>/`. Pass `--run-dir outputs/NAME` to choose a fixed directory. Repeat the command with the same selections to verify and reuse completed jobs. To continue unfinished jobs from their latest checkpoints, add `--resume` and use that same directory.

```sh
python main.py --experiment cold --cases cold_disease --run-dir outputs/cold-disease --resume
```

Open `report.html` after the jobs finish. `summary.json` records the run and aggregate metrics. `metrics.csv` contains the metric summaries and `disease_metrics.csv` contains per-disease results. For fold-based runs, the aggregation first averages each disease over its eligible folds, then computes the macro mean across diseases. Cold cases report their respective evaluation panels.

## CPU smoke runs

Smoke mode uses small synthetic data on CPU, with one default fold or case per suite and the selected methods or variants. The artifacts are marked `SYNTHETIC_TEST_ONLY`.

```sh
python main.py --experiment comparisons --smoke --methods FULL TRAIN_VISIBLE_POPULARITY
python main.py --experiment cold --smoke --cases cold_disease --skip-cold-hgtdr
```

A smoke run that includes matched controls, HGTDR, or TxGNN uses the same corresponding libraries. TxGNN smoke also uses a verified local source checkout or cache. [Native installation](workflow/native_sources/README.md)

## Run the public checks

From the repository root, run the following after the [one-time CPU installation](../README.md#try-the-public-checks).

```sh
python experiments/run_checks.py
```

The command runs the public test suites, checks the five-fold ledger and the ablation source hashes, then trains and scores a small synthetic cold-start case. All fixtures are local or generated during the run. To inspect the output of just the cold-start example, run `python -B experiments/cold_start/synthetic/test_frozen_test_score.py`. It prints the temporary directory containing the checkpoint and `test/metrics.json`.

With just Python, `python experiments/audit/verify_r0.py` checks the published fold ledger and `python experiments/ablations/verify.py sources` checks the published ablation files against their recorded hashes.

## Inputs for the paper experiments

The archived run procedures use the inputs listed below. They preserve the original artifact identities, checkpoints, and analysis records.

| Study | What was evaluated | Inputs for the archived run procedure |
| --- | --- | --- |
| [Baseline comparisons](comparisons/) | Six methods on five R0 folds, including the training-frequency (Popularity) baseline. The [fold ledger](audit/) records each method and output. | Matching research checkout, fold manifests, legal training graphs, features and sampling plans. HGTDR and TxGNN also need their native environments. The reused HGTDR fold needs its selected checkpoint. |
| [Ablations](ablations/) | Paired comparisons of training losses and evidence interventions, plus the earlier six-variant holdout study. | Matching research checkout, frozen schedules, graphs, features and evaluation panels. The pairing audit reads the original update logs. The holdout analysis reads saved predictions and run records. |
| [Cold start](cold_start/) | Drug, disease, double-cold and first-approval-year cohorts. | Prepared research graph and evidence, split and exclusion manifests, and frozen drug and disease features. The approval-year case also needs the drug first-approval-year field. |
| [Cold-disease comparison with HGTDR](cold_start/hgtdr/) | EviHGT and two separately trained HGTDR variants on the same held-out diseases and test candidates. | Prepared cold-disease corpus, completed EviHGT cold-disease run, and the source-bound HGTDR CUDA runtime and feature package. The aggregate results are readable without those inputs. |
| [HGTDR-adapted-contra](hgtdr_contra/) | A separate single-output model fitted on recorded contraindications and evaluated on the five common R0 inner development panels. | Original R0 folds, features and graph caches, matching research modules, and the local HGTDR reconstruction with CUDA. |

The original licensed records, prepared folds, per-pair predictions and trained weights are held outside this repository. To train the portable EviHGT model on your own normalized data, follow the [data guide](../docs/data.md).

To recompute the original R0 comparison without retraining, the [comparison summary script](comparisons/summarize_r0.py) needs the original per-disease result files, their audit index and the recorded HGTDR-adapted score cache. HGTDR-adapted-contra is summarized from its separate five-fold outputs. The [native adapters](native_adapters/) document how HGTDR and TxGNN were integrated.

The [single-fold pilot](pilot/) records the earlier shared support comparison. The [learning-rate study](tuning/) records the five completed FULL settings and how the validation objective determined the selected rate.

[Figures and study results](../docs/experiments.md) · [Bilingual manuscript figures](../docs/manuscript_figures.md) · [Portable training and evaluation](../docs/reproducibility.md)

## Restore the archived source layout

Historical run receipts bind the exact paths and checksums of the sources used at the time. From a Git clone containing the archived revision, the following command restores the earlier public source tree into a new directory.

```sh
python experiments/replay.py --output ../evihgt-replay-source
```

The archive command restores the public source files. To replay an original job with its recorded artifact identities, use its archived procedure and the corresponding inputs.
