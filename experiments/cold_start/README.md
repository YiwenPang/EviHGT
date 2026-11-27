<a id="冷启动实验"></a>

# Cold-start experiments

[中文指南](README.zh-CN.md)

The November 18, 2026 protocol defines four entity holdout cases. Each case starts with a fresh EviHGT FULL model. The earlier support-only cold-drug and cold-disease figures use a separate evaluation protocol. [Experiment figures](../../docs/experiments.md#historical-holdout-evaluation)

<a id="从四个原始文件运行"></a>

## Run from the four source files

Place the four source files as shown in the main README and run these commands from the repository root. The workflow reconstructs each case's graph, frozen features, training plan, and holdout panels. For the approval-year cohort, it reads `properties.year_of_first_approval` from drug records in the OptimusKG 2.0 aggregate node file `nodes.parquet`. [Download and placement](../../README.md#download-and-place-the-four-source-files)

```sh
python main.py --experiment cold
python main.py --experiment cold --cases cold_disease --run-dir outputs/cold-disease
python main.py --experiment cold --smoke --cases cold_disease --skip-cold-hgtdr
```

The default run covers all four cases. On `cold_disease`, it also fits the HGTDR support and contra models and scores them on the same test panel. HGTDR training uses CUDA and `torch-sparse` and `torch-scatter` builds compatible with PyTorch. Add `--skip-cold-hgtdr` to run EviHGT on its own. The third command is a CPU check with synthetic data. [HGTDR installation](../workflow/native_sources/README.md)

Use `--mode prepare` to build the selected cases without training. The run's `report.html` and `summary.json` summarize the results, and `metrics.csv` and `disease_metrics.csv` contain the metric records. After an interruption, use the original options and the same `--run-dir` with `--resume`. [Run directories and recovery](../README.md#run-from-the-source-databases)

<a id="划分与评分"></a>

## Splits and scoring

| Case | Holdout unit |
| --- | --- |
| cold_drug | Canonical drug ID. |
| cold_disease | Canonical disease ID. |
| double_cold | Drug and disease IDs. A pair joins a partition only when both endpoints belong to it. |
| first_approval_year_cohort | Drugs grouped by first approval year. Each year stays in one partition. Drugs without a recorded year are excluded from this case. |

The first three cases hash canonical entity IDs into nominal training, validation, and test proportions of 80%, 10%, and 10%. Positives come from audited development and auxiliary pools. Previously protected pairs remain excluded. The approval-year case uses a static graph and feature snapshot to measure generalization across drug approval cohorts.

Graph construction removes facts, evidence records, and CSR traversals incident to held-out endpoints. It checks both directions and known facts equivalent to the target relations. Endpoints retain the permitted static features, including frozen D1 embeddings and nonclinical drug–gene and disease–phenotype profiles. Upstream encoder pretraining exposure to the target information remains NOT_VERIFIED.

Each case uses seed 42, the frozen FULL configuration, and three training cycles. The trainer receives a role-free exclusion list. A separate integrity check reads the held-out panels, and test role files stay outside the training inventory. Test scoring uses the fixed final-step checkpoint. Within each disease, both heads compare their recorded positives with a shared sample of up to 64 unlabeled drugs. The outputs include per-disease ranking metrics, counts of constant and tied scores, and graph coverage. Each case trains FULL once. The HGTDR comparison uses the same cold-disease split and test panels, with a single training run per model. [HGTDR comparison](hgtdr/)

<a id="源码与运行时身份"></a>

## Source and runtime identity

- [prepare_case.py](prepare_case.py) builds the split, eligible unlabeled candidates, and fixed training plan. [approval_cohort.py](approval_cohort.py) sets the year cutoffs.
- [graph_builder.py](graph_builder.py) writes the masked facts, evidence, and adjacency arrays.
- [run_case.py](run_case.py) checks membership, trains the model, verifies the checkpoint, and scores held-out queries. [run_queue.py](run_queue.py) runs the four cases in sequence.
- [source_sha256.json](source_sha256.json) records LF-normalized SHA-256 hashes of the research scripts, tests, and their initial public copies.

The public adapter resolves workspace paths in locations.py so source and generated files stay separate. Synthetic cold-start tests use the system temporary directory. The original queue checked 130 frozen files, including documentation and figures. The runtime check also uses three complete file inventories. [Workspace paths](locations.py)

| Runtime identity | Inventory | Release contents |
| --- | --- | --- |
| historical-release | [runtime_sha256.json](runtime_sha256.json) | 51 Python source files, two YAML configurations, and the original synthetic fixture. |
| raw-preparation-release | [runtime_sha256_raw_preparation.json](runtime_sha256_raw_preparation.json) | 84 Python source files, three YAML configurations, and the same fixture, for 88 files in total. This release adds raw database preparation. |
| full-workflow-release | [runtime_sha256_full_workflow.json](runtime_sha256_full_workflow.json) | 90 Python source files, four YAML configurations, and the same fixture, for 95 files in total. This release adds final rankings, path explanations, a result report, and a standalone smoke command. |

check_frozen() requires the actual filenames and each LF-normalized hash to match one complete inventory. It rejects added modules, missing or changed files, and mixed releases. Receipts record runtime_variant and runtime_manifest. The two newer identities record historical_runtime_identical as False. The two older inventories retain their original bytes.

raw-preparation-release updates preparation dispatch and missing-corpus messages in the historical CLI. The numerical model and training sources still match the accepted historical hashes. The full-workflow-release runner exports rankings/, explanations/index.json, summary.json, and report.html after the final development evaluation. It also provides main.py --smoke. The model architecture, loss, optimizer updates, three training cycles, and scoring formulas follow the existing implementation. Path explanations read the weights and fact logic from the model's forward pass.

Historical comparisons and the earlier Python AST comparison with docstrings removed and YAML content comparison refer to the original runtime identity. To replay those runs, use the source, inputs, and checkpoint bound in the original receipts. The source restore command retrieves the pinned Git revision. The historical script hashes in source_sha256.json remain unchanged. [Restore the archived source](../README.md#restore-the-archived-source-layout)

run_case.py resolves the case directory before checking containment. Equivalent relative paths and Windows short-path aliases therefore resolve to the same location. The input hash and directory containment checks still apply. The path adapter and its regression test have a separate inventory. This change leaves the model, membership, training budgets, and scoring formulas intact. [Windows path adapter checksums](source_sha256_windows_paths.json)

<a id="软件自检"></a>

## Software checks

Install the repository dependencies and pytest. Run from the repository root with Python 3.11 or later.

    python -B -m pytest experiments/cold_start -q
    python -B experiments/cold_start/synthetic/test_frozen_test_score.py
    python main.py --smoke

The first command checks entity assignments, sampling exclusions, graph masks, membership, score ties, and three runtime identities on synthetic data. The second runs three CPU training updates for a cold case and scores its independent holdout panel.

The third command checks the main workflow with CPU/FP32, 15 synthetic nodes, and three real parameter updates. It saves a development evaluation, checkpoint, rankings, path explanations, and a report. It runs offline with its own synthetic input and configuration. The corpus is written to data/processed/smoke/&lt;uuid&gt;/ and run artifacts to outputs/smoke/&lt;uuid&gt;/&lt;run_id&gt;/. Input and completion records are marked SYNTHETIC_TEST_ONLY. The two smoke commands check their respective software workflows, and their synthetic outputs are outside the paper results.

<a id="running-with-licensed-inputs"></a>
<a id="使用授权数据运行"></a>
<a id="归档研究工作区入口"></a>

## Archived research workspace

The archived commands read derived inputs and hash inventories from the original research workspace. Paths in those inventories must resolve within that workspace. Preparation reads the following artifacts.

- Canonical node index, fact and evidence tables, target map, and graph CSR arrays.
- Development and U manifests, positive memberships, observed and excluded pairs, and static features.
- Verified disease–gene graph, evidence adapter, and graph acceptance record.
- Frozen D1 features. The approval-year case also needs the structured first-approval-year field from the per-type OptimusKG 2.0 drug node Parquet file. [Input contract](../../docs/data.md)

Set the local research workspace and D1 feature paths from the repository root. This example uses PowerShell.

    $env:EVIHGT_RESEARCH_ROOT = (Resolve-Path '../EviHGT-research').Path
    $env:EVIHGT_D1_FEATURES = (Resolve-Path '../HGTDR/data/processed/competition_pilot_v1/features').Path
    python -B experiments/cold_start/run_queue.py

The public repository matches full-workflow-release by default. Set EVIHGT_FROZEN_ROOT to an archived runtime that matches another registered inventory. You can also prepare, validate, and train one case in a new run directory. The HGTDR comparison takes the prepared cold_disease corpus and the completed EviHGT run. [Comparison entry point](hgtdr/)

    python -B experiments/cold_start/prepare_case.py --case cold_drug
    python -B experiments/cold_start/run_case.py --case cold_drug --mode validate
    python -B experiments/cold_start/run_case.py --case cold_drug --mode train

The workspace layout in locations.py matches the paths recorded in the original manifests. Derived inputs are stored by case under CACHE. Training and evaluation records use the same case names under EXPERIMENT/runs. Authorized graph records, membership files, features, predictions, and checkpoints stay local. [Workspace layout](locations.py)
