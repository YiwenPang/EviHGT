# Contributing

Install the appropriate [PyTorch build](README.md#installation) and `requirements-dev.txt`, which includes preprocessing and test dependencies, then run the synthetic CPU suite from the repository root.

```sh
python check_environment.py --device cpu
python -B -m pytest tests -q --basetemp outputs/pytest
```

For changes that affect the published experiment code, also run `python experiments/run_checks.py`. The comparison-ledger check, `python experiments/audit/verify_r0.py`, requires only the Python standard library. These checks use synthetic fixtures and published metadata, so contributors do not need the licensed databases or a prepared research corpus.

Describe the behavior changed and give the commands used to check it. For changes to training or evaluation, identify the affected data fields, model components or metrics and update the corresponding equations and diagrams from the implemented behavior. Keep issue examples small and use synthetic fixtures. Place research inputs and run artifacts in the ignored `data/` and `outputs/` directories; do not commit licensed records or credentials.

Keep data instructions consistent with the [supported preparation routes](docs/data.md#supported-preparation-routes). `main.py` prepares an absent corpus from the raw configuration before a fresh training run; `--mode prepare` stops after corpus validation. The published raw recipe supports R0F1 and verifies its membership, mask and sampling identities. `check_data.py` checks placement, OptimusKG identities and DrugBank headers without preparing a corpus. `prepare_data.py` imports normalized inputs, and `export_data.py` migrates a completed research corpus with its sealed plan.

Use synthetic inputs to test these interfaces, preserve source files and check each command's prerequisites. For preprocessing changes, test source validation, target/reverse/equivalent masking, stage-hash reuse, interruption handling and offline cache behavior as applicable. Keep pinned resource contracts versioned. Changed build identities require a new stage directory and final corpus destination; never rewrite completed receipts to make changed inputs appear reusable. Keep original result records distinct from new software checks or preparation runs.

Maintain independent support and contraindication-related channels. Use verified relation fields for labels and keep unknown values distinct from explicit negatives. If a change alters the data partitions, sample order, or held-out exclusions, assign the resulting corpus a new identity. When reporting a metric, specify its positive and unlabeled candidate pools. For ablations, name the computation removed or replaced.

Use English by default in source comments and primary documentation. Keep English and Chinese paper figures synchronized, with labels in the language of each version and shared axis terms retained in both. Preparation currently prints Chinese phase messages. Console messages should identify the phase, preserve method IDs and metric names, and point to the full error log when a run fails.

Maintain every project-owned README in both English and Chinese. Use `README.md` as the English default, with a short `中文指南` link near the top to the complete `README.zh-CN.md`; add an `English` link back on the Chinese page. Update both versions together, keeping steps, commands, options, filenames, defaults, output paths, versions and necessary limitations consistent. Check links and section anchors in both languages. This rule excludes third-party dependencies, caches and generated directories.
