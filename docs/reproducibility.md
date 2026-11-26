# Reproducibility guide

`main.py` prepares an absent corpus from the configured raw databases, validates it and trains the FULL model on its fixed three-cycle schedule. The default corpus is `data/processed/corpus/manifest.json`. The [preparation guide](data.md#supported-preparation-routes) covers raw preparation, existing portable corpora and explicit research exports.

To reproduce a recorded result, use the original code, data, splits, features, seed, update budget, and evaluation panels for that run. The [software verification](#software-verification) section describes checks that run with synthetic fixtures and the published result ledger, without the research datasets.

## Before a run

1. Record the source version and its commit or release identifier when available. Retain a copy or checksum of the configuration used for the run.
2. Select a Python environment and run `python check_environment.py --device cuda` or `--device cpu`. Keep the resulting report with the run records.
3. If obtaining the raw databases, follow the [download and placement guide](data.md#download-and-place-the-source-databases), preserve the original files, and run `python check_data.py`. This read-only, standard-library check verifies the OptimusKG file identities and the DrugBank XML/XSD headers. It does not build a corpus or validate all source records.
4. For raw preparation, install `requirements-preprocess.txt` and run `python main.py --mode prepare`, adding `--raw-config /path/to/raw.yaml` for another source configuration. This builds the original R0F1 corpus without training. Missing pinned Open Targets and encoder files are downloaded automatically; use `--offline` with complete caches. Alternatively, supply a complete portable corpus, import verified normalized artifacts with `prepare_data.py`, or export a completed research cell with `export_data.py`.
5. Preserve the corpus manifest and every bound SHA-256. The fixed schedule determines sample order, shared global U, conditional local U and fallback behavior. Point the YAML `manifest` setting to this corpus if it is outside the default location.
6. Run `python main.py --mode validate`, adding `--config /path/to/run.yaml` when using another configuration. Validation loads the prepared inputs and writes its report to a new run directory under the configured `output_dir`, before model creation or training. Check evidence and relation semantics against the source audits.
7. Save both YAML configurations when using raw preparation. `feature_device` and `feature_batch_size` belong to the raw configuration; training `device` and `precision` belong to the run configuration. Record both devices and the dependency versions.

The [experiment index](../experiments/) links the published results to their study-specific code, local inputs and verification commands. For a run whose receipts record its original source layout, see [how to restore the archived sources](../experiments/README.md#restore-the-archived-source-layout).

## Raw recipe and preparation identity

The raw entry point implements the original R0F1 recipe. It reconstructs the historical candidate exclusions and holdout reservations before the five-fold and inner reservations, then verifies the resulting membership and mask hashes. Its three complete positive cycles preserve the original global U order and the conditional U records for cycles 2 and 3. Checkpoint hashes retained in the seed recipe are deterministic constants; preparation does not load those checkpoints or recreate historical training receipts.

OptimusKG aggregate file identities are fixed to release 2.0. DrugBank XML/XSD identities and schema information are recorded from the local authorized files. Open Targets 25.06 datasource files and encoder revisions/files have separate pinned resource contracts. Retain these identities with the final corpus, because changing structured data or features changes the experiment even when the same command still runs.

Completed stages under `stage_root` have file-hash receipts. Rerunning the same preparation verifies those files before reuse and preserves failed attempts. Changed source bytes, code, bound package versions or feature settings require a new `stage_root` and a new final `manifest` destination. Explicit preparation checks an existing final corpus against the available raw sources and its source, recipe and build identity before reuse. Use `--mode validate` when checking a corpus without raw sources. Normal training with an existing corpus skips raw preparation. See [preparation recovery](data.md#resume-preparation-or-rebuild-changed-inputs) for the exact behavior.

## Training and ablations

`main.py` runs one three-cycle FULL plan. A fresh training invocation prepares missing inputs before entering that plan; validation, checkpoint evaluation and checkpoint resume require the existing corpus. `ablation.py` runs explicitly selected variants serially. All variants consume the same corpus and plan and use the same initialization seed. The common nominal parameter schema is retained even when a branch is bypassed; report effective computation as well as nominal parameter count.

Running `main.py` does not launch all paper folds, baseline comparisons, cold-start studies or final-test evaluations. Those studies retain their own protocols and required inputs in the experiment directories. The published result tables come from the original research runs.

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
- Raw configuration, `stage_root/identity.json`, stage receipts, and `output_dir/preparation/<attempt-id>/` reports when preparing from raw files.
- Hardware, dependency versions, precision, seed, variant and update budget.
- Parent checkpoint when resuming, output directory, final checkpoint and development metrics.
- Measured elapsed time and a precise statement of which stages it includes.
- Deviations, failures, exclusions and unresolved source questions.

Keep raw data, derived restricted inputs, checkpoints and predictions local under the configured data/output directories. Public reports can describe schemas and reproducibility identities without redistributing licensed records.

## Software verification

From the repository root, `python experiments/audit/verify_r0.py` checks the published five-fold ledger using only the Python standard library. After installing the [CPU test dependencies](../README.md#try-the-public-checks), run `python experiments/run_checks.py` for the public model and experiment checks. These commands do not need the raw databases or a prepared research corpus. Passing them verifies the exercised software and published ledger; reproducing the paper's scores requires the matching research inputs listed in the [experiment guide](../experiments/README.md#inputs-for-the-paper-experiments).

The 46 tests in the initial release passed in the source directory and after relocation under an isolated Python invocation. Tests use synthetic fixtures and cover corpus schemas and hashes, label exclusions, direct/reverse target removal, fixed conditional records, feature masks, model gradients and ablations, metrics, console output and error handling.

The comparisons with the research implementation recorded:

- Exact equality of all 132 model state tensors after mapping the pretrained-fusion module name.
- Exact CPU forward tensors and active gradients in eight train/eval, empty/nonempty-path and zero/nonzero-D1 combinations on one CPU thread.
- Matching global and conditional PU losses and logit gradients for full/partial fallback, global-U reuse and local-U cases, including 5,312 pairs across the 4,096-pair reduction boundary.
- Matching inputs for 282 tensors across six synthetic training/development queries.
- No numerical-body changes across 225 shared definitions and constants in a source comparison.
- Exact final CPU model and AdamW state after an injected interruption at update 1 and resumption through update 3. The integration test also covers graph assembly, all three cycles, development scoring and checkpoint evaluation.

The equivalence comparisons above apply to the stated CPU settings and fixtures. [Recorded development results](experiments.md) and [instrumented runtime components](runtime_reference.json) come from the original research workflow.

On 26 November 2026, `main.py --mode prepare` completed from the four original database files in a newly installed, independent environment. The run used verified public Open Targets and encoder caches in offline mode. It generated every canonical, split, cleaned-graph, feature and sampling artifact afresh and passed the complete portable-corpus loader before publication. No previously prepared research corpus, feature arrays or trained checkpoint served as an input to that build.

The final R0F1 corpus contains 202,131 nodes, 14,709,000 facts, 14,726,957 evidence rows and 26,756,592 traversals across 82 bound files. All seven membership hashes, the exclusion mask, three cycle hashes and all 13,694 conditional records matched the original recipe. The 56 fact/evidence tables, role-free target map, forbidden-fact table and CSR arrays matched the independently verified historical reconstruction. Recomputed SideInfo and D1 arrays also matched the recorded research features byte for byte in the tested CUDA environment. Canonical-stage counts and traversal-order checks are documented in [raw preparation](raw_preparation.md).

The same independent environment passed all 131 core tests and the complete public experiment-check command, including synthetic three-update training, checkpoint recovery and cold-start scoring. A separate network check downloaded one Open Targets shard and both encoder configuration files from their pinned official URLs and verified their sizes and SHA-256 values. These checks validate the new preparation workflow and its software integration. The paper's training runs and reported evaluation scores remain the original experiments.

An additional read-only check loaded the newly generated corpus, assembled one real training query with 1,000 nodes and 5,000 traversals, and ran the seeded, untrained FULL model in inference mode. Both heads returned finite tensors of shape `[1]`; model parameters and the corpus manifest were unchanged. This query exercised the no-path fallback and a missing drug D1 feature. Nonempty paths and the other feature branches are covered by the synthetic model tests. The check made no optimizer updates and produced no experimental performance result.

GitHub Actions runs the synthetic tests on Linux and Windows with Python 3.11.
