# Implemented model

`build_model` in `src/evi_hgt/model.py` initializes EviHGT from an `EviHGTConfig`, a seed and frozen pretrained features. It returns an object with `.model`, `.forward(batch, path_batch=...)` and `.metadata()`. To reload a saved `state_dict`, rebuild the model with the same configuration, variant and seed so the corresponding ablation hooks are restored. See [recorded development results](experiments.md).

## Node and evidence representations

The node representation starts with a type embedding. Available drug and disease raw features pass through distinct linear, LayerNorm, and GELU branches and combine with the type representation through a vector-valued sigmoid gate. The packed binary input has 3,588 dimensions: 2,048 molecular fingerprint bits, 512 drug profile bits, two drug availability bits, 512 disease ontology bits, 512 phenotype bits, and two disease availability bits. Missing feature families contribute no values. Other node types retain their type representation.

Frozen 768-dimensional drug and disease representations enter before query encoding and graph reasoning through separate bias-free projections. These projections start at zero and are trainable. The availability mask sets the contribution of missing vectors to zero throughout training and evaluation.

Evidence records encode numeric values and masks, Boolean values and masks, property tokens, evidence tokens, provenance tokens, and relation identity. Three independent attention pools summarize all records, verified support records, and verified counter records. The all-record pool retains context independent of polarity.

Two distinct strength functions produce support strength `s` and counter strength `c`. Where both channels are semantically available, the fact masses are:

```text
T = s (1 - c)
F = (1 - s) c
B = s c
U = (1 - s) (1 - c)
```

Facts without both semantic channels carry a zero vector and a false availability mask. An available fact can have a high `U` mass. Edge context combines relation, all/support/counter pools, provenance, available masses, and six availability/observation indicators.

## Graph and path computation

The graph encoder uses typed multi-head attention, query-conditioned edge context and Pre-LN residual blocks. For available facts, `FULL` assigns fixed weights of 0.25 to each T/F/B/U source-value transform. Facts without available logic use a separate `NA` transform. Learned fact masses enter the edge context and affect attention and messages through it.

The graph readout concatenates final drug and disease embeddings, their elementwise product, their absolute difference, and the query embedding constructed before graph propagation. A projection produces a shared query representation.

Bounded paths are encoded by the path Transformer, scored against the query, and pooled within each query. A scalar sigmoid gate adds a projected path residual to the graph readout. Empty path sets produce an exact graph-only fallback. The model requires an explicit packed path batch, including when it is empty.

Independent support and contra heads map the shared representation to two logits. Their sigmoid outputs define predicted query T/F/B/U masses with the same product formulas above. The two heads yield independent, uncalibrated scores. `ranking_score` is the predicted `T` mass. Channel-specific evaluation uses the corresponding head logit.

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
