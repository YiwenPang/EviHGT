# Contributing

Install the appropriate [PyTorch build and test dependencies](README.md#try-the-public-checks), then run the synthetic CPU suite from the repository root.

```sh
python check_environment.py --device cpu
python -B -m pytest tests -q --basetemp outputs/pytest
```

For changes that affect the published experiment code, also run `python experiments/run_checks.py`. The comparison-ledger check, `python experiments/audit/verify_r0.py`, requires only the Python standard library. These checks use synthetic fixtures and published metadata, so contributors do not need the licensed databases or a prepared research corpus.

Describe the behavior changed and give the commands used to check it. For changes to training or evaluation, identify the affected data fields, model components or metrics and update the corresponding equations and diagrams from the implemented behavior. Keep issue examples small and use synthetic fixtures. Place research inputs and run artifacts in the ignored `data/` and `outputs/` directories; do not commit licensed records or credentials.

Keep data instructions consistent with the [supported preparation routes](docs/data.md#supported-preparation-routes). `check_data.py` checks raw-file placement, OptimusKG identities and DrugBank headers without changing files. `prepare_data.py` packages already normalized inputs, and `export_data.py` requires a completed research corpus and sealed sampling plan. The raw-to-training preprocessing entry point is not yet published. Do not describe raw downloads as sufficient to run training. Use synthetic inputs when checking changes to these interfaces, preserve source files, and verify any new commands against their actual prerequisites.

Maintain independent support and contraindication-related channels. Use verified relation fields for labels and keep unknown values distinct from explicit negatives. If a change alters the data partitions, sample order, or held-out exclusions, assign the resulting corpus a new identity. When reporting a metric, specify its positive and unlabeled candidate pools. For ablations, name the computation removed or replaced.

Use English by default in source comments and primary documentation. Keep English and Chinese paper figures synchronized, with labels in the language of each version and shared axis terms retained in both. Public-release console output uses English. Console messages should identify the current phase, show training progress, and point to the full error log when a run fails.
