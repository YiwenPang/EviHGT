# Six-method comparison on the R0 development folds

This directory contains the research code behind the common drug–disease ranking comparison. Each of five R0 folds has one inner development panel for therapeutic support and another for contraindications. Methods scoring the same task in a fold receive the same drug–disease pairs and P/U labels. P denotes a recorded positive for that task. Both panels use the same sampled unlabeled pairs (U). The five folds overlap and belong to a single repeat.

## What each method received

| Method | Inputs and training | Checkpoint used for development scoring |
| --- | --- | --- |
| EviHGT (FULL) | Query graph and candidate paths, node side information, frozen D1 features and fact-level evidence | End of three training cycles |
| HGT-D1 | PyG HGT on the query graph, with the same node side information and D1 features | Same endpoint, sampling schedule, loss and optimizer as FULL |
| PairMLP-D1 | Drug and disease node features, with the same node encoder and D1 feature source | Same endpoint, sampling schedule, loss and optimizer as FULL |
| HGTDR-adapted | HGTDR with its fixed node features and query-masked sampler, trained for therapeutic support | Best inner support ROC-AUC over 300 epochs, then AP, then the earlier epoch |
| TxGNN-adapted | One epoch of relation pretraining and 500 epochs of DDA fine-tuning | 20 inner support evaluations. The selected model scores both tasks |
| Popularity | Independent counts of the unique training-positive diseases for each drug and task | Fixed score of `log1p(count)` for all candidate diseases. Zero for drugs with no training positives in that task |

The three matched models used one update per positive training group and a common three-cycle budget of 20,421–20,568 updates per fold. Feature inputs, training budgets and selection opportunities differ across the two native adaptations and the matched models. The native comparisons therefore describe complete systems on the same development candidates. Native checkpoint selection used the inner support panel on which performance is reported. The loss used by FULL and the matched controls was also chosen during development, with R0F0 and R0F1 among the earlier loss comparisons. [Ablation and loss experiments](../ablations/)

The comparison covers 30 method–fold combinations. The queue trained 19 new model instances, verified existing endpoints for five FULL folds and one HGTDR support fold, and scored Popularity in each fold. HGTDR contributed one support score per candidate in this comparison. The subsequent contraindication-specific training experiment has its own code and selection record. [HGTDR contraindication experiment](../hgtdr_contra/README.md)

## From training membership to the published table

1. `TrainingInputs` constructs each training view. The corpus builder removes blocked clinical facts and their evidence, and binds the eligible graph to the fold. The two native receivers also validate the allowed fact IDs before graph export. [Training membership](src/evi_hgt_ranking_comparison/data.py) · [Graph construction](src/evi_hgt_ranking_comparison/corpus.py) · [Native receivers](src/evi_hgt_ranking_comparison/native_runtime.py)
2. The matched models consume the same ordered P/U sampling plan, optimizer schedule and loss. HGT-D1 and PairMLP-D1 share their endpoint readout initialization. FULL supplies its own graph and path computation. The training frequency baseline derives its two count dictionaries directly from training positives. [Sampling](src/evi_hgt_ranking_comparison/sampling.py) · [Shared training](src/evi_hgt_ranking_comparison/training.py) · [Matched models](src/evi_hgt_ranking_comparison/models.py) · [FULL](src/evi_hgt_ranking_comparison/full.py) · [Popularity preparation](src/evi_hgt_ranking_comparison/popularity_preparation.py)
3. The coordinator reads the exact run manifest and dispatches one method–fold job at a time. Native workers store the selected weights separately from their final recovery state. For the reused HGTDR fold R0F0, the retained epoch-222 weights produced the scores. [Training entry point](scripts/ranking_comparison/runtime.py) · [Coordinator](src/evi_hgt_ranking_comparison/runtime.py) · [Native checkpoint handling](src/evi_hgt_ranking_comparison/native_checkpoint.py) · [Reused weights](src/evi_hgt_ranking_comparison/reuse.py)
4. Scoring uses the development membership and checks the candidate and checkpoint identities. ROC-AUC and AP group exactly tied scores. Top-K metrics use canonical drug ID to break ties and cap K at the available candidate count. A disease needs at least one P and one U. For the R0 table, metrics are averaged over the eligible folds of each disease, then equally across diseases. [Development scoring](src/evi_hgt_ranking_comparison/runtime_evaluation.py) · [Metric definitions](src/evi_hgt_ranking_comparison/metrics.py) · [R0 table calculation](summarize_r0.py)

Under this estimator, support AP was 0.873292 for FULL, 0.835220 for the matched HGT and 0.901453 for HGTDR-adapted. Within this original six-method comparison, HGT-D1 had the highest contra AP (0.897898), and Popularity had the highest contra ROC-AUC (0.932036) and NDCG@10 (0.934880). Averaging folds equally gives a different weighting and places Popularity first for contra AP in this group. The expanded [main table](../../README.md#development-results) also includes the later HGTDR-adapted-contra model, whose contra AP is 0.942237. These measurements concern recorded positives versus sampled unlabeled candidates, and all checkpoint and aggregation rules should accompany comparisons. [Figures and results](../../docs/experiments.md)

## Source identity and running the checks

The source inventory pairs each of the 45 named files with its SHA-256 in this release, its original SHA-256 in the November 9 execution inventory, and the corresponding Git blob. The protocol identifiers that enter sampling, weight initialization, and saved metric and count records retain their original values. [Source hashes](SOURCE_SHA256.json) · [Protocol identifiers](src/evi_hgt_ranking_comparison/protocol_ids.py)

From the repository root, the synthetic CPU tests exercise metric ties, P/U handling, candidate membership errors and training-only frequency counts. They require Python and pytest.

```sh
python -m pytest -o pythonpath=experiments/comparisons/src experiments/comparisons/tests/ranking_comparison
```

With access to the private research results, the summary can be recomputed without loading a model. Pass the R0 audit JSON and the selected HGTDR R0F0 score cache. This historical fold records the candidate identities as individual rows. The utility checks the cache checksum and recomputes the membership digest from those rows.

```sh
python experiments/comparisons/summarize_r0.py /path/to/metrics_audit.json /path/to/epoch_0222/validation.ROWS.json
```

The named training source shows how the research runs constructed graphs, trained the methods, and selected checkpoints. The source uses descriptive aliases for research packages, storage paths, receipt fields and checkpoint parameter names. Historical manifests and checkpoints retain their original names and checksums. Each named training entry verifies its original research script against the registered digest and runs that script in the supplied research checkout. The coordinator applies the original manifest, permission and source checks. For example, this command prints the coordinator's arguments.

```sh
python experiments/comparisons/scripts/ranking_comparison/runtime.py --research-root /path/to/research --help
```

To obtain the matching source files and entry points from a Git checkout, run the following command from the repository root. Choose a new output directory.

```sh
python experiments/replay.py --output NEW_DIRECTORY
```

The restored comparison README lists the original entry points. Full training also requires the matching private research checkout, its registered fold manifests, graph and feature caches, sampling plans, native HGTDR and TxGNN environments, and job permissions. Redistribution restrictions apply to parts of these inputs. Upstream target exposure during D1 pretraining remains unverified. [Release data contract](../../docs/data.md) · [Native adapters](../native_adapters/) · [Earlier single-fold pilot](../pilot/)
