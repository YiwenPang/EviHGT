# Experiment code

## Run the public checks

From the repository root, run the following after the [one-time CPU installation](../README.md#try-the-public-checks).

```sh
python experiments/run_checks.py
```

The command runs the public test suites, checks the five-fold ledger and the ablation source hashes, then trains and scores a small synthetic cold-start case. All fixtures are local or generated during the run. To inspect the output of just the cold-start example, run `python -B experiments/cold_start/synthetic/test_frozen_test_score.py`. It prints the temporary directory containing the checkpoint and `test/metrics.json`.

With just Python, `python experiments/audit/verify_r0.py` checks the published fold ledger and `python experiments/ablations/verify.py sources` checks the published ablation files against their recorded hashes.

## Inputs for the paper experiments

The [database download and placement guide](../docs/data.md#download-and-place-the-source-databases) gives source links, filenames and a directory tree. The study inputs below also include the prepared graph, features and fixed splits used by each experiment.

| Study | What was evaluated | Inputs needed to repeat the run |
| --- | --- | --- |
| [Baseline comparisons](comparisons/) | Six methods on five R0 folds, including the training-frequency (Popularity) baseline. The [fold ledger](audit/) records each method and output. | Matching research checkout, fold manifests, legal training graphs, features and sampling plans. HGTDR and TxGNN also need their native environments. The reused HGTDR fold needs its selected checkpoint. |
| [Ablations](ablations/) | Paired comparisons of training losses and evidence interventions, plus the earlier six-variant holdout study. | Matching research checkout, frozen schedules, graphs, features and evaluation panels. The pairing audit reads the original update logs. The holdout analysis reads saved predictions and run records. |
| [Cold start](cold_start/) | Drug, disease, double-cold and first-approval-year cohorts. | Prepared research graph and evidence, split and exclusion manifests, and frozen drug and disease features. The approval-year case also needs the drug first-approval-year field. |
| [Cold-disease comparison with HGTDR](cold_start/hgtdr/) | EviHGT and two separately trained HGTDR variants on the same held-out diseases and test candidates. | Prepared cold-disease corpus, completed EviHGT cold-disease run, and the source-bound HGTDR CUDA runtime and feature package. The aggregate results are readable without those inputs. |
| [HGTDR-adapted-contra](hgtdr_contra/) | A separate single-output model fitted on recorded contraindications and evaluated on the five common R0 inner development panels. | Original R0 folds, features and graph caches, matching research modules, and the local HGTDR reconstruction with CUDA. |

The original licensed records, prepared folds, per-pair predictions and trained weights are held outside this repository. To train the portable EviHGT model on your own normalized data, follow the [data guide](../docs/data.md).

To recompute the original R0 comparison without retraining, the [comparison summary script](comparisons/summarize_r0.py) needs the original per-disease result files, their audit index and the recorded HGTDR-adapted score cache. HGTDR-adapted-contra is summarized from its separate five-fold outputs. The [native adapters](native_adapters/) document how HGTDR and TxGNN were integrated.

The [single-fold pilot](pilot/) records the earlier shared support comparison. The [learning-rate study](tuning/) records the five completed FULL settings and how the validation objective determined the selected rate.

[Figures and study results](../docs/experiments.md) · [Portable training and evaluation](../docs/reproducibility.md)

## Restore the archived source layout

Historical run receipts bind the exact paths and checksums of the sources used at the time. From a Git clone containing the archived revision, the following command restores the earlier public source tree into a new directory.

```sh
python experiments/replay.py --output ../evihgt-replay-source
```

This command restores the public source files. Model training and scoring are separate steps that use the inputs listed above.
