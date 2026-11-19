# Experiment code

| Paper experiment | What to inspect |
| --- | --- |
| [Baseline comparisons](comparisons/) | Five-fold R0 comparison of six methods, including the training-frequency (Popularity) baseline. The [fold ledger](audit/) records the candidates and scores used for each method and output. |
| [Ablations](ablations/) | Paired comparisons of training losses and evidence interventions, plus the earlier six-variant holdout study. The study guide links to each loss, sampling schedule and model intervention. |
| [Cold start](cold_start/) | Split construction, leakage checks, training and scoring for the drug, disease, double-cold and first-approval-year cohorts. |

The original five-fold HGTDR comparison evaluated therapeutic support. The [HGTDR-contra study](hgtdr_contra/) fitted a separate model on recorded contraindications. The [native adapters](native_adapters/) document how HGTDR and TxGNN were integrated.

The [single-fold pilot](pilot/) records the earlier shared support comparison. The [learning-rate study](tuning/) records the five completed FULL settings and how the validation objective determined the selected rate.

[Figures and study results](../docs/experiments.md) · [Portable training and evaluation](../docs/reproducibility.md)

## Replaying a recorded run

Historical run receipts bind the exact paths and checksums of the sources used at the time. From a Git clone containing the archived revision, the following command restores the earlier public source tree into a new directory.

```sh
python experiments/replay.py --output ../evihgt-replay-source
```

The replay contains public source files. Each study guide lists the local inputs needed for a full rerun, including licensed records, frozen splits, features and checkpoints where applicable.
