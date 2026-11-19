# R0 comparison audit

The [fold ledger](r0_fold_audit.csv) records all 55 method/fold/output results behind the original six-method comparison. It includes aggregate counts, the SHA-256 of sorted (drug, disease, P/U role) triples, the number of evaluable diseases, and three fold-level metrics. Each row identifies the original metric and completion files by checksum. HGTDR R0F0 used an earlier cache whose metric header omitted the panel checksum. Its digest in the ledger was recomputed from the saved candidates at selected epoch 222.

Run the public consistency check from the repository root.

```sh
python experiments/audit/verify_r0.py
```

With the original local results tree, the following command also checks the artifact digests, validates candidate identities, and recomputes fold-level macro ROC-AUC, AP and NDCG@10 from the saved per-disease metrics. The files with per-candidate native scores allow the verifier to reconstruct their panel digests.

```sh
python experiments/audit/verify_r0.py --research-root /path/to/original/EviHGT
```

The verifier uses the local R0 metrics audit when it can identify a single copy. You can specify that file with `--artifact-index PATH`. If the index is missing or ambiguous, a Git clone containing the archived source revision can recover the original artifact locations from the recorded ledger. In all cases the published file checksums and panel identities apply.

## How to interpret the comparison

- Within each fold, scoring uses the same candidates for each available output. A positive is a recorded relation of the appropriate type. U denotes a sampled unlabeled pair. The source filters direct, reverse and equivalent target facts. Upstream exposure of the pretrained encoders to target information remains unverified.
- FULL, HGT-D1 and PairMLP-D1 share node inputs, sampling, loss and a fixed update budget. They differ in architecture and effective computation. FULL has higher support AP than both matched models on all five folds.
- Adapted HGTDR selects among 300 epochs using inner support ROC-AUC, then AP. Adapted TxGNN has one pretraining epoch and 500 fine-tuning epochs, with 20 support selection opportunities. Its contra scores use the same support-selected weights. These are system comparisons on the development panels used for selection. Adapted HGTDR leads support AP in the original comparison. The separately contraindication-trained HGTDR adapter uses its own selection and result lineage.
- Popularity scores the training-positive frequency of each drug separately for each output. It leads contra AUC and NDCG@10, and matched HGT-D1 leads contra AP under the main aggregation. Fold-first averaging also puts Popularity first on contra AP. Candidate composition and repeated disease coverage matter for interpreting this task.
- The main results average each disease across its eligible folds and then weight diseases equally. The 55 rows here are fold summaries. Recomputing the main table uses the disease-level observations. The five folds overlap and support descriptive comparisons.

[Reported main table](../../README.md#development-results) 路 [Source for the comparison](../comparisons/)
