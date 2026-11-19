# Contributing

Install the appropriate PyTorch build and the packages in `requirements-dev.txt`, then run the synthetic CPU suite.

```sh
python check_environment.py --device cpu
python -B -m pytest tests -q --basetemp outputs/pytest
```

Describe the behavior changed and give the commands used to check it. For changes to training or evaluation, identify the affected data fields, model components or metrics and update the corresponding equations and diagrams. Keep issue examples small and use synthetic fixtures. Place research inputs and run artifacts in the ignored `data/` and `outputs/` directories.

Maintain independent support and contraindication-related channels. Use verified relation fields for labels and keep unknown values distinct from explicit negatives. If a change alters the data partitions, sample order, or held-out exclusions, assign the resulting corpus a new identity. When reporting a metric, specify its positive and unlabeled candidate pools. For ablations, name the computation removed or replaced.

Use English in source comments, documentation, figure labels and console output. Console messages should identify the current phase, show training progress, and point to the full error log when a run fails.
