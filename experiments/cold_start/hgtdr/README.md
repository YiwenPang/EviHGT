# HGTDR cold-disease comparison

[中文指南](README.zh-CN.md)

The comparison uses the strict disease split prepared by the [cold-start scripts](../). Two single-output HGTDR models learn from recorded indications and contraindications, respectively. Each uses the native 300-epoch recipe with AdamW at a starting rate of 0.003, cosine decay and equal numbers of recorded positives and sampled unlabeled pairs in each batch. Checkpoint selection uses development disease-macro ROC-AUC, then AP, then the earlier epoch. The recorded support and contra checkpoints were selected at epochs 149 and 53, respectively.

The selected models rank the same held-out candidates as EviHGT. The disease is absent from training and its incident graph facts are removed. Both methods use the same frozen drug and disease embeddings. EviHGT also uses its static side information and reports its fixed three-cycle checkpoint. [Results and paired intervals](../../../docs/experiments.md#cold-disease-comparison)

## Read the results

From the repository root, use Python 3.11 or later. The aggregate report is available without model or data dependencies.

```sh
python -B -m experiments.cold_start.hgtdr.run summary
```

## Repeat training and scoring

The paper run uses the audited EviHGT research workspace and its sibling HGTDR reconstruction described in the [native adapter guide](../../native_adapters/). Use the interpreter that provides the native HGTDR implementation, CUDA and compatible PyTorch, PyG and torch-sparse packages. It also needs the frozen HGTDR feature package and its identity receipts. The native runtime uses the same workspace and feature bindings as the [R0 contra adapter](../../hgtdr_contra/).

With a local authorized dataset, prepare and run the `cold_disease` case as described in the [cold-start guide](../#running-with-licensed-inputs). Pass the directory containing its prepared `manifest.json` and the directory containing the completed EviHGT `result.json`. The following checks the inputs and native graph without training:

```sh
python -B -m experiments.cold_start.hgtdr.run validate --research-root /path/to/research --corpus /path/to/prepared/cold_disease
```

Train both models and score the test candidates in one command:

```sh
python -B -m experiments.cold_start.hgtdr.run run --research-root /path/to/research --corpus /path/to/prepared/cold_disease --reference-run /path/to/evihgt/runs/cold_disease
```

Use `train` to stop after training and checkpoint selection. Use `score` with the same corpus and run directory and pass `--reference-run` to score the test candidates after both training receipts have been committed. The default graph cache and output paths are `data/cache/cold_disease_hgtdr` and `outputs/cold_disease_hgtdr` under the research workspace. Use `--cache` and `--output` to select other directories within those parent directories. An interrupted training run resumes from the latest compatible epoch checkpoint when invoked with the same settings. After both heads finish, `result.json` reports the paired comparison.

[Recorded aggregate results](results.json) · [Source checksums](source_sha256.json) · [Data sources and access](../../../docs/data.md)
