# Implemented model

`build_model` in `src/evi_hgt/model.py` initializes EviHGT from an `EviHGTConfig`, a seed and frozen pretrained features. It returns an object with `.model`, `.forward(batch, path_batch=...)` and `.metadata()`. To reload a saved `state_dict`, rebuild the model with the same configuration, variant and seed so the corresponding ablation hooks are restored. See [recorded development results](experiments.md).

## Node and evidence representations

The node representation starts with a type embedding. Available drug and disease raw features pass through distinct linear, LayerNorm, and GELU branches and combine with the type representation through a vector-valued sigmoid gate. The packed binary input has 3,588 dimensions: 2,048 molecular fingerprint bits, 512 drug profile bits, two drug availability bits, 512 disease ontology bits, 512 phenotype bits, and two disease availability bits. Missing feature families contribute no values. Other node types retain their type representation.

Frozen 768-dimensional drug and disease representations enter before query encoding and graph reasoning through separate bias-free projections. These projections start at zero and are trainable. The availability mask sets the contribution of missing vectors to zero throughout training and evaluation.

Evidence encoding first represents individual source records, then pools records belonging to the same canonical fact. Each record encodes numeric values and masks, Boolean values and masks, property tokens, evidence tokens, provenance tokens, and relation identity. Three independent attention pools summarize all records, verified support records, and verified counter records. The all-record pool retains context independent of polarity.

Two separately parameterized strength functions produce support strength `s` and counter strength `c` (the refutation strength denoted by `r` in the manuscript). Each function reads its own pooled vector and observation flag. An available channel with no records can still produce a latent strength from the zero pooled vector, observation flag and learned bias. Where both channels are semantically available, the fact masses are:

```text
T = s (1 - c)
F = (1 - s) c
B = s c
U = (1 - s) (1 - c)
```

These four components sum to one for an available fact. `B` represents concurrent learned support and counter strengths; `U` represents weak strengths in both channels. Separate observation flags record whether either kind of evidence was present. Facts without both semantic channels carry a zero vector and a false availability mask (`NA`). Edge context combines the relation embedding, the all-record pool, availability-masked support/counter pools, mean provenance, available masses, and six availability/observation indicators. Graph messages and path steps retrieve this shared context by canonical fact identifier.

In the supplied input implementation, fact polarity applies to verified anatomy-to-gene expression relations. `EXPRESSION_PRESENT` and `EXPRESSION_ABSENT` retain separate canonical facts because their relation names differ. Their support/refutation semantics concern the expression proposition. At the clinical ranking level, the support and contra heads instead learn from their respective indication and contraindication positives.

## Graph and path computation

The graph encoder uses typed multi-head attention, query-conditioned edge context and Pre-LN residual blocks. For available facts, `FULL` assigns fixed weights of 0.25 to each T/F/B/U source-value transform. Facts without available logic use a separate `NA` transform. Learned fact masses enter the edge context and affect attention and messages through it.

The graph readout concatenates final drug and disease embeddings, their elementwise product, their absolute difference, and the query embedding constructed before graph propagation. A projection produces a shared query representation.

Each bounded path step joins its final endpoint embeddings with the shared fact context. The path Transformer encodes the step sequence; step attention normalizes over valid steps within a path, and query-conditioned path attention normalizes over candidate paths for the same query. A single scalar sigmoid gate per query adds the projected pooled path context to the graph readout. Empty path sets set the gate to zero and produce an exact graph-only fallback. The model requires an explicit packed path batch, including when it is empty.

A path weight gives that path's share in the pooled representation. To assess its effect on candidate ranking, remove or replace the path for the same query and compare recomputed scores and ranks. The weight itself records pooling behavior; interpreting a path as a biological mechanism requires external evidence.

Separate support and contra heads map the shared representation to two logits. Each head learns the ordering of its own recorded positives and unlabeled candidates, so the two scores can vary separately. Their sigmoid outputs define predicted query T/F/B/U masses with the same product formulas above. `ranking_score` is the predicted `T` mass; channel-specific evaluation uses the corresponding head logit.

The half-softplus ranking objective depends on positive-versus-unlabeled logit differences. Adding the same constant to every logit from one head leaves these differences and the loss unchanged, but changes its sigmoid scores and the query masses. Clinical probability interpretation therefore requires independent calibration and validation. See the [model card](model_card.md) for output interpretation and the status of upstream pretraining exposure.

## Available variants

| Variant | Change from FULL |
| --- | --- |
| `FULL` | Complete architecture. |
| `NO_D1` | Bypass pretrained-vector residuals. |
| `NO_SIDE` | Bypass gated raw side-information fusion; keep type embeddings and D1. |
| `NO_PATH` | Bypass the path module and path residual. |
| `CONTENT_ZERO` | Zero evidence-text and provenance content blocks and the shared provenance output; keep properties, relations, and masks. |
| `FACT_QUALITY_CONSTANT` | Use four equal fact masses where available; preserve unavailable masks and other edge-context fields. |

All variants share the nominal parameter schema and initialization. A bypassed branch receives no active gradient. [Recorded development results](experiments.md#historical-two-cell-ablations) include the two-cell content and fact-quality ablations.

## Implementation verification

The 10-node-type, 57-edge-type, 128-hidden-dimension FULL configuration has 3,846,610 parameters. The tests cover isolated initialization, finite dual-head gradients, content and side-information suppression, missing-feature masks after learning, constant fact-quality masks and rejection of missing required inputs. See [the reproducibility guide](reproducibility.md#software-verification) for equivalence and recovery checks and their limits.
