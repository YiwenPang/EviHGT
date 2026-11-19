# Native baseline adaptations

The five-fold and pilot comparisons use local implementations of two published systems. The paths below preserve the original sibling checkout names. The [Phase98 comparison](../comparisons/) controls fold membership, the legal graph and scoring in the five-fold study.

| System | Upstream source | Local adaptation |
| --- | --- | --- |
| HGTDR | [Pinned author repository](https://github.com/bcb-sut/HGTDR/tree/a0adec64b08c9d70835e1e7c04a513b9be6a87db) | [Fold adapter](EviHGT/src/evi_hgt/phase9/hgtdr_adapter.py), [sampling, training and scoring](HGTDR/src/competition.py), [feature inputs](HGTDR/src/competition_features.py) |
| TxGNN | [Pinned author repository](https://github.com/mims-harvard/TxGNN/tree/0e0203c3939f7d12da27aaf10e55daaff5424023) | [Training-only sampling](EviHGT/src/evi_hgt/phase9/txgnn_adapter.py), [training recipe and selection](EviHGT/src/evi_hgt/phase9/txgnn_protocol.py), [model integration](TxGNN/txgnn/competition.py) |

The HGTDR reconstruction uses frozen BioBERT names and ChemBERTa molecular features, with missing-input handling. It uses three HGT layers of width 64 with eight heads, mean relation aggregation, concatenation of layer outputs and a batch-normalized MLP. The predictor dropout is 0.2. The native loss retains the upstream per-example positive weighting. The local notebook-derived `HGTDR/src/runtime.py` is an external prerequisite. Its recorded SHA-256 is `583c40f389a8395881703b8661bc1097150f4f36ca1cdbaac159b85d9f2b2fe8`. The bundled PyG compatibility layer carries its [license](HGTDR/src/vendor/PYG_LICENSE). The HGTDR-contra study fits the same single-output architecture on a different target.

TxGNN uses the upstream HeteroRGCN and DistMult model. Its disease prototype keys and degrees use fold-visible training positives. The original `model.py` is available at the pinned commit. The [compatibility patch](TxGNN/utils-compat.patch) replaces pandas DataFrame append calls with concatenation in the upstream utilities. Apply it at the pinned TxGNN repository root using `git apply /path/to/utils-compat.patch`. [TxGNN license](TxGNN/LICENSE)

The [source checksums](source_sha256.txt) identify the archived project adapters. The original training commands require the named repository layouts, native packages, graph and feature caches, and the frozen fold and selection records. [Earlier pilot protocol](../pilot/) · [Separate contraindication adapter](../hgtdr_contra/)
