# EviHGT model card

This card describes the intended use, input assumptions and output interpretation of `FULL`. See the [architecture](model.md), [recorded development results](experiments.md) and [software checks](reproducibility.md#software-verification).

## Task and intended use

EviHGT ranks canonical drug-disease pairs using a biomedical knowledge graph, evidence properties and provenance, entity features, and bounded biological paths. It predicts two independent channels:

| Channel | Observed positive supervision | Interpretation |
| --- | --- | --- |
| Support | Verified `INDICATION` pairs admitted by the upstream inclusion policy | Relative support for an indication-related association. |
| Contra | Verified `CONTRAINDICATION` pairs | Relative evidence for a contraindication-related association. |

Use EviHGT to study association ranking, inspect the edges and paths used in scoring, and test model components. Attention weights and extracted paths describe the computation. The scores have no prospective clinical validation and are unsuitable for patient-level predictions, dosing or treatment decisions.

## Inputs and implementation scope

Training requires a locally prepared portable corpus: canonical node/fact/evidence tables, a physically masked CSR graph, source-audited supervision and exclusions, raw SideInfo, frozen D1 vectors with availability masks, explicit development panels, and the supplied three-cycle plan. See the [data contract](data.md) for schemas and preparation commands.

The model uses independent support/counter evidence pools, differentiable fact logic, two query-conditioned heterogeneous Transformer layers, and a gated path residual. In `FULL`, the four direct logic-message transforms receive uniform weights of 0.25 when logic is available. Learned fact masses affect edge context and subsequent attention/messages indirectly; the four direct transform weights remain fixed. Unavailable facts use a separate `NA` transform.

The local extractor retains at most two hops, 1000 nodes and 5000 messages per query. Path search retains up to 16 paths of length 2-4. The returned paths depend on these search bounds and the supplied graph.

## T/F/B/U and availability

For independent learned support strength `s` and counter strength `c`, the available fact representation is:

| State | Mass | Meaning within the representation |
| --- | --- | --- |
| `T` | `s * (1 - c)` | Support without strong counter evidence. |
| `F` | `(1 - s) * c` | Counter evidence without strong support. |
| `B` | `s * c` | Support and counter evidence coexist. |
| `U` | `(1 - s) * (1 - c)` | Neither learned channel is strong. |
| `NA` | No normalized mass vector; availability is false | The fact is outside the verified two-channel semantic scope, or that scope is unavailable. |

`T/F/B/U` are latent masses learned from evidence. For `NA`, all four stored masses are zero and the availability mask is false. An available fact can have a high `U` mass. Separate flags mark whether support or counter evidence was observed.

In this input implementation, direct two-channel fact logic is available for verified ANA-to-GEN `EXPRESSION_PRESENT`/`EXPRESSION_ABSENT` relations. Other relation families supply context and messages through the `NA` transform. See [source semantics and labeling](data.md#source-map-and-manifest).

The two query prediction heads also produce T/F/B/U values using the same product formulas after sigmoid transformation. These masses aggregate the two query scores. They sum to one and are uncalibrated. See [the fact-logic implementation](../src/evi_hgt/models/four_valued_logic.py) and [prediction outputs](../src/evi_hgt/training/prediction.py).

## Scores and training objective

The model exposes support and unsuitability logits, their sigmoid scores, predicted query masses, and a `ranking_score` equal to predicted `T` mass. Use the corresponding head logit to reproduce a channel-specific metric.

Training uses the half-softplus positive-unlabeled ranking loss on positive-versus-U logit differences. The supplied plan preserves one global U per original group. The first cycle uses global comparisons; later support comparisons use supplied same-disease U records with an explicit fallback, while contra retains the original global U stream.

U denotes unlabeled candidate pairs. The loss trains relative ordering. The two sigmoid scores vary independently, and the workflow uses no calibrated decision threshold.

## Evaluation interpretation

The local workflow evaluates only the supplied development panels. Support compares verified indication P with validation U. Contra compares verified contraindication P with that same validation U pool. Both use the corresponding independent head.

ROC-AUC, AP, trapezoid AUPR and top-K metrics quantify separation of observed positives from U in a fixed candidate panel. Their interpretation depends on candidate selection and the positive associations that remain unlabeled.

Metrics are computed by disease and macro-averaged over diseases with both P and U. Missing classes are reported as skips; when nothing is evaluable, the console shows `N/A`. AP uses step integration, while trapezoid AUPR uses adjacent exact-score threshold endpoints; the two can differ substantially when scores are tied. Rankings use deterministic drug-ID tie breaking and `effective_k = min(K, candidate_count)`. In small panels, `Hits@K` may be guaranteed by the candidate counts. Report those counts, skipped diseases, the metric definition and candidate identity together with any score. See [metrics.py](../src/evi_hgt/metrics.py).

See [recorded ablations](experiments.md#historical-two-cell-ablations) for the two-cell comparison.

## Leakage and provenance boundaries

The corpus must already remove held-out target facts and their evidence physically. The loader checks split exclusions and requires target mappings covering direct/reverse drug-disease facts regardless of relation name. Each query additionally removes its target-equivalent fact IDs before bounded extraction. Resolve semantically equivalent relations during preparation and include them in the map.

Observed clinical pairs are excluded from U draws. Evaluation members are blocked from training, and graph assembly for evaluation receives role-free queries. Source hashes, immutable group order, conditional records and feature availability are checked. Audit source assertions and entity mappings during corpus preparation.

D1 uses frozen drug SMILES and canonical disease-name representations from the pinned ChemBERTa/BioBERT sources documented in [data.md](data.md). Clinical label text, supervised DDA features, trained node embeddings and prediction scores are excluded by the input contract. Upstream pretrained-encoder exposure to target-related knowledge is `NOT_VERIFIED`.

## Known limits and reproducibility

- Coverage, missingness, source redundancy and inconsistent annotations can affect evidence representations and ranking.
- Bounded neighborhood/path selection depends on the supplied graph and physical traversal order. Changing those inputs changes the model's evidence context.
- Prepare source-audited inputs that match the published contract or export a completed research corpus.
- CPU/FP32 and CUDA/BF16 are separate execution settings. The equivalence tests cover the documented CPU configuration.
- The historical [runtime summary](runtime_reference.json) covers instrumented components of the research runs.

For a local experiment, retain the corpus identity, source audits, candidate panels, variant, seed, configuration, environment snapshot, checkpoint and raw metric artifacts. Follow the source-data terms when retaining or sharing derived artifacts.
