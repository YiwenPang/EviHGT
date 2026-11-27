# Training objective and ablation experiments

[中文指南](README.zh-CN.md)

The main entry point runs the paired loss and evidence experiments. This directory also records the earlier six-variant holdout study, which used its own models, data splits, and checkpoint rule.

## Run from the four source files

Use the source files and preparation settings from the main guide. Run from the repository root. [Download and placement](../../README.md#download-and-place-the-four-source-files)

```sh
python main.py --experiment ablations
python main.py --experiment losses
```

`ablations` runs `FULL`, `CONTENT_ZERO`, and `FACT_QUALITY_CONSTANT` on `R0F1` and `R1F0`. `losses` runs FULL with `softplus` and `sigmoid` on `R0F0`, `R0F1`, and `R1F0`. It uses the dedicated `DEV95` recipe for the `R0F0` loss comparison.

```sh
python main.py --experiment ablations --cells R0F1 --variants FULL FACT_QUALITY_CONSTANT
python main.py --experiment losses --cells R0F0 --mode prepare
python main.py --experiment losses --smoke
```

`--mode prepare` stops after input preparation. Smoke mode runs on synthetic CPU data and defaults to one fold. Open `report.html` in the experiment's run directory for the summary and use `disease_metrics.csv` for the per-disease scores. [Run directories and checkpoint recovery](../README.md#run-from-the-source-databases)

## Paired loss and evidence experiments

The complete comparison uses three development folds for loss selection and two for the evidence interventions. Within a fold, the loss arms started from the same initialization with independent optimizer state. They used the same node features, graph masks, positive and unlabeled samples, update schedule and development candidates. Each update log records the sample identities and numbers of pairs for both output channels. Conditional unlabeled drugs were drawn uniformly from the eligible training universe. The frozen checkpoint digest helped set the draw seed. The sampler did not read model scores.

| Fold | Updates per arm | Comparisons |
| --- | ---: | --- |
| R0F0 | 20,568 | Bounded sigmoid vs. half-softplus loss |
| R0F1 | 20,541 | Both losses, FULL vs. content zero, FULL vs. constant quality |
| R1F0 | 20,700 | Both losses, FULL vs. content zero, FULL vs. constant quality |

Read the [loss definitions](development/src/evi_hgt_loss_comparison/losses.py), [first-fold driver](development/src/evi_hgt_loss_comparison/runner.py), [two-fold driver](development/src/evi_hgt_paired_losses/runner.py) and [paired schedule](development/src/evi_hgt_paired_losses/sampling.py). The [ablation driver](development/src/evi_hgt_evidence_ablation/runner.py) reuses the half-softplus update and checks nine sampling fields against FULL at every logical step. Its [model hooks](development/src/evi_hgt_evidence_ablation/model.py) use the [original intervention code](development/src/evi_hgt/ranking_protocol/model.py).

- **Content zero** sets the pooled evidence content and provenance vectors to zero. The remaining input fields and model structure are retained. Effective gradient flow changes. AP differences vary by fold and output channel.
- **Constant quality** sets the four evidence masses to 0.25 at eligible positions and retains the availability mask. Relative to this variant, FULL increased contraindication AP by 0.92 and 0.09 percentage points in R0F1 and R1F0. Support AP was about 0.11 points lower in each fold.

These comparisons use the fixed three-cycle endpoint on the inner development panels. The R1F0 sigmoid run repeated 116 logged updates after a checkpoint recovery. Matching uses the logical update number and checks the sample identities of repeated records. Physical compute differed. The recovery excerpts show the original [replay accounting and worker](development/scripts/paired_losses/recovery/recovery_excerpts.py) and the [continuation and reporting code](development/scripts/evidence_ablation/recovery/recovery_excerpts.py). The latter reused one completed intervention and finished the three remaining arms.

## Historical six-variant study

The [training driver](historical/src/evi_hgt/training/formal_driver.py), [variant catalog](historical/src/evi_hgt/models/variants.py) and [supplemental variant](historical/src/evi_hgt/protocols/confirmatory_ablation.py) describe FULL, the project's relation-only BASE_HGT, and four ablations. The [candidate protocol](historical/src/evi_hgt/evaluation/evaluable_protocol.py), [scoring driver](historical/src/evi_hgt/evaluation/overnight_runtime.py) and [post-completion analysis](historical/scripts/analyze_results.py) show how all four evaluation stages were constructed and scored.

Stage A compared support scores on the same frozen set of 47,769 candidate pairs for all six methods. Of 1,354 diseases, 1,109 had both a recorded positive and an unlabeled candidate. Tied scores receive half credit for ROC-AUC and are grouped by threshold for AP. Top-K ties use ascending canonical drug ID. A five-rate validation sweep for FULL selected a learning rate of 0.00005. This rate was then used for all arms. The sweep gave FULL additional tuning opportunities. Each arm used its own best validation checkpoint under the same rule. BASE selected step 9,380 and the other five arms selected step 18,760. Architecture and parameter counts differed.

FULL reached a macro AP of 0.475181 compared with 0.412357 for BASE_HGT. All four ablations exceeded FULL on AP in this study. The result describes one training seed and the sampled support task. Stage B evaluated FULL alone on six prespecified diseases using the full candidate universe. Stages C and D evaluated FULL alone on the historical cold-drug and cold-disease support panels. Their macro ROC-AUC values were 0.4950 and 0.4992. The [historical cold visibility policy](historical/src/evi_hgt/data/cold_visibility.py) allowed certified side information for the query endpoint at scoring time.

## Verification

From the repository root, check the published source bytes with the Python standard library.

```sh
python experiments/ablations/verify.py sources
```

The same command accepts `--research-root PATH` to verify the original file hashes and the run receipts against an existing research workspace. With the private update logs available, the following command also checks all three loss comparisons and all four intervention runs. It verifies complete logical step coverage, the nine pairing fields and repeated sample identities.

```sh
python experiments/ablations/verify.py pairing --research-root PATH
```

The historical analysis includes synthetic cases for tied rankings, skipped queries, paired bootstrap resampling, exact metric recomputation and candidate membership. With NumPy, Polars, tqdm and pytest installed, run the tests from this directory.

```sh
cd historical
python -m pytest -q -o pythonpath=src tests/test_results_analysis.py
```

From the repository root, run the protocol regression tests with the following command. They use a synthetic three-cycle example to check the schedule digest, conditional sampling bindings, random seeds and selected candidates against values from the original implementation. They also check the three fold-specific initialization seeds and the fusion-layer checkpoint keys.

```sh
python -m pytest -q experiments/ablations/tests/test_protocol_ids.py
```

[SHA256.json](SHA256.json) records the published, earlier release and original research checksums for each file. Two recovery modules contain excerpts. The historical study's original file layout, receipt mappings and exact source bindings are available in the [replay archive](../README.md).

The [protocol identifiers](development/src/ablation_protocol_ids.py) preserve the exact strings used in the original random seeds, schedule hashes, feature metadata and model parameter names. The [artifact identifiers](development/src/ablation_artifacts.py) and [record identifiers](development/src/ablation_records.py) name the corresponding diagnostic panels, checkpoints, manifests and run receipts. Public directory and script names describe the experiment each implements.

The named modules expose the scientific implementation for review. The historical records bind file paths, field names and exact source digests. To run one of the research training, registration or evaluation scripts against its original workspace, pass `--research-root PATH` alongside the script's original arguments. The launcher finds the corresponding research script in the pinned archive, checks its SHA-256 and executes it with the current Python interpreter in the research workspace. The original dependency, selection and authorization checks apply there. The public synthetic analysis tests and published source checks run locally.
