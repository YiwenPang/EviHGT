# Reproducibility guide

The portable runner executes a specified corpus, schedule and configuration. See the [data contract](data.md) for how to prepare those inputs. To reproduce a recorded result, use the original code, data, splits, features, seed, update budget, and evaluation panels for that run. The [software verification](#software-verification) section covers the synthetic tests and implementation comparisons.

## Before a run

1. Record the source version and its commit or release identifier when available. Retain a copy or checksum of the configuration used for the run.
2. Select a Python environment and run `python check_environment.py --device cuda` or `--device cpu`. Keep the resulting report with the run records.
3. Preserve source database versions and original files. Prepare a corpus with verified entity mappings, clinical relation semantics, deduplication, feature sources, and leakage exclusions.
4. Preserve the corpus manifest and every bound SHA-256. The fixed schedule determines sample order, shared global U, conditional local U and fallback behavior.
5. Validate the corpus with `python main.py --mode validate`. Check evidence and relation semantics against the source audits.
6. Save the exact YAML configuration. Record device, precision and dependency versions with the run.

The [experiment index](../experiments/) links the published results to their study-specific code, local inputs and verification commands. For a run whose receipts record its original source layout, see [source replay](../experiments/README.md#replaying-a-recorded-run).

## Training and ablations

`main.py` runs the supplied three-cycle FULL plan. `ablation.py` runs explicitly selected variants serially. All variants consume the same corpus and plan and use the same initialization seed. The common nominal parameter schema is retained even when a branch is bypassed; report effective computation as well as nominal parameter count.

For an ablation comparison, keep data membership, sample sequence, feature availability, optimizer settings, total updates, candidate panels and metric definitions fixed. Record the exact variant ID.

Checkpoints record model parameters, AdamW state, RNG states and the logical update. A resumed invocation writes a new run directory and records both the loaded update and the number of newly executed updates. Preserve the parent checkpoint and input identities when describing recovery. Load trusted local checkpoints only.

## Evaluation

The supplied development panels determine which pairs are scored. Support compares verified indication positives against U. Contra compares verified contraindication positives against the same development U pool. U is the pool of unlabeled candidates.

Record the candidate membership hash, positive and U counts, evaluable/skipped disease counts, raw head logits, and the exact checkpoint. Report AP and trapezoid AUPR under their respective definitions. For Top-K metrics, report effective K and candidate-pool sizes. Small pools can guarantee Hits independently of model quality.

Match saved predictions to the checkpoint, variant, candidates and split that produced them. Keep unavailable metrics marked `N/A`. See [score interpretation](model_card.md#evaluation-interpretation).

## What to retain locally

Retain the following with each run:

- Code identity and configuration.
- Source versions, corpus identity, feature provenance and plan identity.
- Hardware, dependency versions, precision, seed, variant and update budget.
- Parent checkpoint when resuming, output directory, final checkpoint and development metrics.
- Measured elapsed time and a precise statement of which stages it includes.
- Deviations, failures, exclusions and unresolved source questions.

Keep raw data, derived restricted inputs, checkpoints and predictions local under the configured data/output directories. Public reports can describe schemas and reproducibility identities without redistributing licensed records.

## Software verification

The 46 tests in the initial release passed in the source directory and after relocation under an isolated Python invocation. Tests use synthetic fixtures and cover corpus schemas and hashes, label exclusions, direct/reverse target removal, fixed conditional records, feature masks, model gradients and ablations, metrics, console output and error handling.

The comparisons with the research implementation recorded:

- Exact equality of all 132 model state tensors after mapping the pretrained-fusion module name.
- Exact CPU forward tensors and active gradients in eight train/eval, empty/nonempty-path and zero/nonzero-D1 combinations on one CPU thread.
- Matching global and conditional PU losses and logit gradients for full/partial fallback, global-U reuse and local-U cases, including 5,312 pairs across the 4,096-pair reduction boundary.
- Matching inputs for 282 tensors across six synthetic training/development queries.
- No numerical-body changes across 225 shared definitions and constants in a source comparison.
- Exact final CPU model and AdamW state after an injected interruption at update 1 and resumption through update 3. The integration test also covers graph assembly, all three cycles, development scoring and checkpoint evaluation.

The equivalence comparisons above apply to the stated CPU settings and fixtures. [Recorded development results](experiments.md) and [instrumented runtime components](runtime_reference.json) come from the original research workflow.

GitHub Actions runs the synthetic tests on Linux and Windows with Python 3.11.
