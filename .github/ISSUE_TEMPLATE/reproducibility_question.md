---
name: Data or reproducibility question
about: Ask about raw inputs, corpus preparation, model outputs or a recorded result.
title: "[Reproducibility] "
---

## Question

Which field, equation, command, metric or recorded run are you asking about?

## Context

Which workflow are you using?

- [ ] Public software checks (`experiments/run_checks.py` or `experiments/audit/verify_r0.py`)
- [ ] Raw-file placement and identity check (`check_data.py`)
- [ ] Raw preparation (`main.py --mode prepare` or automatic preparation before training)
- [ ] Prepared input import or research export (`prepare_data.py` or `export_data.py`)
- [ ] Corpus validation, training or checkpoint evaluation (`main.py` or `ablation.py`)
- [ ] Interpretation or reproduction of a published result

Include the commit, operating system, Python version, exact command and a short error or log excerpt. Add the relevant details below:

- For raw inputs: dataset release, extracted filenames, and the result of `check_data.py` if run. Remove private directory names from its output.
- For preparation: the last stage, online/offline mode, preprocessing package versions, `cell`, `feature_device`, `feature_batch_size`, and relevant messages from `outputs/preparation/<attempt-id>/`. State whether the final manifest already existed and whether the code, inputs or environment changed since the previous attempt.
- For prepared inputs: the preparation route, manifest schema, and corpus or schedule identity if available. State whether `main.py --mode validate` passed.
- For model results: variant, split, checkpoint or update, support/contra channel, and candidate counts.

The [data guide](https://github.com/YiwenPang/EviHGT/blob/main/docs/data.md) lists source versions, filenames, installation requirements and recovery steps. `check_data.py` checks placement. `main.py` prepares missing R0F1 inputs and trains; `--mode prepare` prepares and validates without training. The [public checks](https://github.com/YiwenPang/EviHGT#try-the-public-checks) run without research datasets.

Use synthetic IDs and a minimal example. Keep licensed records, raw databases, checkpoints and private credentials local; share only the configuration or log excerpts needed to explain the question.
