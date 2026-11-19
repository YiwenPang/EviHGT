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

1. `TrainingInputs` constructs each training view. The corpus builder removes blocked clinical facts and their evidence, and binds the eligible graph to the fold. The two native receivers also validate the allowed fact IDs before graph export. [Training membership](src/evi_hgt_phase98/data.py) · [Graph construction](src/evi_hgt_phase98/corpus.py) · [Native receivers](src/evi_hgt_phase98/native_runtime.py)
2. The matched models consume the same ordered P/U sampling plan, optimizer schedule and loss. HGT-D1 and PairMLP-D1 share their endpoint readout initialization. FULL supplies its own graph and path computation. The training frequency baseline derives its two count dictionaries directly from training positives. [Sampling](src/evi_hgt_phase98/sampling.py) · [Shared training](src/evi_hgt_phase98/training.py) · [Matched models](src/evi_hgt_phase98/models.py) · [FULL](src/evi_hgt_phase98/full.py) · [Popularity preparation](src/evi_hgt_phase98/popularity_preparation.py)
3. The coordinator reads the exact run manifest and dispatches one method–fold job at a time. Native workers store the selected weights separately from their final recovery state. For the reused HGTDR fold R0F0, the retained epoch-222 weights produced the scores. [Historical entry point](scripts/phase98/runtime.py) · [Coordinator](src/evi_hgt_phase98/runtime.py) · [Native checkpoint handling](src/evi_hgt_phase98/native_checkpoint.py) · [Reused weights](src/evi_hgt_phase98/reuse.py)
4. Scoring uses the development membership and checks the candidate and checkpoint identities. ROC-AUC and AP group exactly tied scores. Top-K metrics use canonical drug ID to break ties and cap K at the available candidate count. A disease needs at least one P and one U. For the R0 table, metrics are averaged over the eligible folds of each disease, then equally across diseases. [Development scoring](src/evi_hgt_phase98/runtime_evaluation.py) · [Metric definitions](src/evi_hgt_phase98/metrics.py) · [R0 table calculation](summarize_r0.py)

Under this estimator, support AP was 0.873292 for FULL, 0.835220 for the matched HGT and 0.901453 for HGTDR-adapted. For contra, HGT-D1 had the highest AP (0.897898), and Popularity had the highest ROC-AUC (0.932036) and NDCG@10 (0.934880). Averaging folds equally gives a different weighting and places Popularity first for contra AP. These measurements concern recorded positives versus sampled unlabeled candidates, and all checkpoint and aggregation rules should accompany comparisons. [Figures and results](../../docs/experiments.md)

## Source identity and running the checks

The 45 files under `src/`, `scripts/` and `tests/` retain the same paths and bytes as the selected files in the November 9 execution source inventory. The SHA-256 manifest records their original digests. The separate `summarize_r0.py` release utility recomputes the final table from saved per-disease results and checks fold membership before aggregation. [Source hashes](SOURCE_SHA256.json)

From the repository root, the original synthetic CPU tests exercise metric ties, P/U handling, candidate membership errors and training-only frequency counts. They require Python and pytest.

```sh
python -m pytest -o pythonpath=experiments/comparisons/src experiments/comparisons/tests/phase98
```

With access to the private research results, the summary can be recomputed without loading a model. Pass the R0 audit JSON and the selected HGTDR R0F0 score cache. This historical fold records the candidate identities as individual rows. The utility checks the cache checksum and recomputes the membership digest from those rows.

```sh
python experiments/comparisons/summarize_r0.py /path/to/metrics_audit.json /path/to/epoch_0222/validation.ROWS.json
```

Full training uses the original EviHGT research checkout and its matching historical `evi_hgt.phase9`, `evi_hgt.phase85*`, training, model, and `evi_hgt_dev*` packages, including the D1 feature loader. It also requires the original registered per-fold manifests, graph and feature caches, sampling plans, run permissions, host-specific job control, and the native HGTDR and TxGNN environments. Those research inputs include data with redistribution restrictions. The public CPU tests and the historical training entry point have different dependencies. Upstream target exposure during D1 pretraining remains unverified. [Release data contract](../../docs/data.md) · [Native adapters](../native_adapters/) · [Earlier single-fold pilot](../pilot/)
