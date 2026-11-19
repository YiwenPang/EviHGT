# Cold-start experiments

These scripts prepare and evaluate four entity holdouts using a fresh EviHGT FULL model in each case. They implement the November 18, 2026 protocol. The earlier support-only cold-drug and cold-disease figures in the [experiment gallery](../../docs/experiments.md#historical-holdout-evaluation) used a separate evaluation protocol.

## Splits and scoring

| Case | Held-out unit |
| --- | --- |
| `cold_drug` | Canonical drug ID |
| `cold_disease` | Canonical disease ID |
| `double_cold` | Both IDs. A pair belongs to a partition only when both endpoints belong to it. |
| `first_approval_year_cohort` | Drug, grouped by first approval year. Whole years stay together. Drugs without a recorded year are excluded. |

The first three cases hash the relevant canonical entity IDs into train, validation and test partitions with nominal shares of 80%, 10% and 10%. The positive pairs come from the audited development and auxiliary pools. Previously protected pairs remain excluded. The approval-year experiment uses static graph and feature snapshots to measure transfer across approval cohorts.

The graph builder removes facts incident to a held-out endpoint, along with their evidence records and CSR traversals. Both edge directions and known target-equivalent facts are checked. Endpoints retain permitted static features, including pretrained D1 inputs and nonclinical drug–gene or disease–phenotype summaries. Possible exposure in upstream encoder pretraining remains unverified.

Each case uses seed 42, the frozen FULL configuration and three training cycles. The trainer has access to role-free block lists. A separate integrity check reads the held-out panels, and their test-role files stay outside the trainer's input inventory. Test scoring uses the fixed last-step checkpoint. Each head ranks its recorded positive pairs against the same set of up to 64 sampled unlabeled drugs per disease. Results include per-disease ranking metrics, counts of constant scores and ties, and graph coverage. A single FULL run per setting measures its holdout performance. Comparisons to other models and variability across seeds require additional runs.

## Source files

- [prepare_case.py](prepare_case.py) builds the split, legal unlabeled candidates and fixed training schedule. [approval_cohort.py](approval_cohort.py) defines the year boundaries.
- [graph_builder.py](graph_builder.py) materializes the masked facts, evidence and adjacency.
- [run_case.py](run_case.py) checks memberships, trains, verifies the checkpoint and scores the held-out queries. [run_queue.py](run_queue.py) launches the four cases serially.
- [source_sha256.json](source_sha256.json) records the SHA-256 of each script or test as used in the research directory and of the public copy with LF line endings. The source directory was `outputs/phase9/phase98/cold_start_20261118/`.

The published adapters replace workstation-specific paths with [locations.py](locations.py), separate the location of source scripts from runtime outputs and use the system temporary directory for the synthetic smoke run. The original queue verified its entire frozen 130-file distribution, including documentation and figures. The public check pins all 51 Python runtime sources, both configurations and the synthetic data fixture. It accepts the original or published source spelling with LF-normalized SHA-256. Python syntax trees with docstrings removed and parsed YAML contents matched across the frozen and public versions of these files. The accepted digests are in [runtime_sha256.json](runtime_sha256.json).

## Running the software checks

Use Python 3.11 or later with the repository dependencies and pytest. Run these checks from the public repository root.

```sh
python -B -m pytest experiments/cold_start -q
python -B experiments/cold_start/synthetic/test_frozen_test_score.py
```

The first command checks entity assignment, sample exclusions, graph masking, memberships and ties on invented inputs. It also checks that the public runtime source pin rejects a changed file or extra module. The second performs a three-update CPU training run and held-out evaluation on a tiny synthetic corpus.

## Running with licensed inputs

Use a separate research workspace with the original, hash-bound derived inputs and their manifests. The paths in the source inventories must resolve in that workspace. Preparation reads the following inputs.

- The canonical node index, fact/evidence components, target map and graph CSR files under `data/processed/phase3/v1/`.
- The development and U manifests, positive memberships, observed and forbidden pairs, and static features under `data/processed/phase85_part2/`.
- The certified disease–gene graph and evidence adapter under `data/processed/phase85_part2b/`, with `outputs/phase85_part2b/graph_acceptance.json`.
- The frozen D1 feature package and, for the fourth case, the OptimusKG 2.0 drug node parquet with its structured first-approval-year field. [Input contract](../../docs/data.md)

From a checkout of this repository, set the paths for the local research workspace and D1 features. The following example uses PowerShell.

```powershell
$env:EVIHGT_RESEARCH_ROOT = (Resolve-Path '../EviHGT-research').Path
$env:EVIHGT_D1_FEATURES = (Resolve-Path '../HGTDR/data/processed/competition_pilot_v1/features').Path
python -B experiments/cold_start/run_queue.py
```

The published checkout supplies the frozen runtime by default. Set `EVIHGT_FROZEN_ROOT` to use a separately archived matching runtime. A single case can also be prepared, validated and trained in a fresh run directory.

```sh
python -B experiments/cold_start/prepare_case.py --case cold_drug
python -B experiments/cold_start/run_case.py --case cold_drug --mode validate
python -B experiments/cold_start/run_case.py --case cold_drug --mode train
```

Prepared inputs are stored under `data/cache/phase98/cold_start_20261118/`. Per-case training and evaluation receipts are written under `outputs/phase9/phase98/cold_start_20261118/runs/` in the research workspace. Licensed graph records, membership files, features, predictions and checkpoints are local run inputs or outputs.
