# Experiment implementations

| Study | Code and scope |
| --- | --- |
| Six-method development comparison | [Five-fold R0 comparison](comparisons/) covers the matched HGT and pair MLP, training-frequency baseline, and native baseline adaptation. The original HGTDR entry evaluated therapeutic support. |
| Domain baselines | [Native adapters](native_adapters/) document the local HGTDR and TxGNN interfaces. [HGTDR-contra](hgtdr_contra/) separately fits the single-output HGTDR model on recorded contraindications. |
| Evidence and training objective ablations | [Ablations](ablations/) covers the paired evidence interventions, loss comparison and earlier six-arm study. |
| Held-out entities | [Cold start](cold_start/) includes the separate drug, disease, double-cold and first-approval-year cohort protocols. Each case evaluates one FULL run. |
| Earlier baseline comparison | [October pilot](pilot/) contains the one-fold, one-seed support comparison and its scoring protocol. |
| Historical tuning | [Learning-rate selection](tuning/) includes the five completed FULL runs, their validation objectives and the selection and reporting code. |
| Audit of the reported comparison | [Five-fold audit](audit/) contains one aggregate row per method, fold and output, the identities of the common candidate panels, and a read-only verifier. |

The [result figures](../docs/experiments.md) identify which study produced each plot. The [portable runner](../docs/reproducibility.md) provides a separate interface for training on a prepared corpus. Each study directory identifies the archived code, any changes made for publication, and the local inputs needed to repeat the original run. The original folds and their licensed inputs, per-pair predictions and model weights stay in the research workspace.
