# October common development pilot

[中文指南](README.zh-CN.md)

The pilot evaluated HGTDR, TxGNN and an existing EviHGT-side checkpoint on a shared masked graph and the same support candidates. One fold and one seed were used. The two native systems each tried three learning rates. HGTDR used CUDA and scored the development panel after each of 300 epochs. TxGNN used CPU, one pretraining epoch and 500 fine-tuning epochs, with development selection every 25 epochs. The cached EviHGT-side result came from an earlier development selection. Their native losses, inputs and model selection budgets differed. Observed support ROC-AUC ranked HGTDR first, followed by EviHGT and TxGNN.

The [pilot settings](settings.json) preserve the recorded scientific parameters with workstation paths removed. The package tree under [EviHGT](EviHGT/) groups the pilot by data export, scoring and execution. [Data export](EviHGT/src/evi_hgt/competition/data.py) constructs the common graph, candidates and sampling policy. [Protocol](EviHGT/src/evi_hgt/competition/protocol.py) checks membership and selects candidates. The [driver](EviHGT/src/evi_hgt/competition/runner.py) checks each baseline's receipts, checkpoint identity, saved scores and metrics. The corresponding [native adapters](../native_adapters/) contain the baseline-side code.

Both baseline jobs completed. A change to an unrelated dispatch branch in the workspace entry point made the parent driver's final source check fail. The [public entry point](EviHGT/main.py) exposes the pilot commands. The [source replay](../replay.py) restores the full dispatch table used at launch. The subsequent audit rechecked all six learning-rate candidates and the cached control. The parent run retains its failed status.

The [checksums](source_sha256.txt) identify the published files. The published copies include layout and wording edits. Their [original checksums](archived_sha256.json) identify the versions in the replay. Fully replaying the original `--competition` command requires the historical research packages, prepared common graph, local model checkouts, environments, run protocol and checkpoint that supplied the control.
