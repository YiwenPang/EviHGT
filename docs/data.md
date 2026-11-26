# Data preparation and portable input contract

`main.py` prepares the original R0F1 corpus from local OptimusKG and DrugBank files when the configured corpus manifest is absent, then trains the FULL model, evaluates the development panels and writes rankings, path explanations and an HTML report. The corpus contains canonical entity, fact and evidence tables, a masked CSR graph, fixed entity features, verified clinical memberships and a three-cycle sampling plan. Its default manifest is `data/processed/corpus/manifest.json`. The [English](../README.md) and [Chinese](../README.zh-CN.md) guides cover the complete run.

## Download and place the source databases

Run commands from the repository root, the folder containing `main.py`. The paths below match the original research inputs.

```text
data/
├── drugbank_all_full_database.xml
├── drugbank.xsd
└── cache/optimuskg/doi_10_7910_DVN_IYNGEV/2.0/recovery/aggregate/
    ├── nodes.parquet
    └── edges.parquet
```

### OptimusKG

Open [OptimusKG release 2.0 on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IYNGEV&version=2.0), download the aggregate `nodes.parquet` and `edges.parquet`, and place them in the directory shown above. Use release **2.0**, not the latest release or the per-type files under `nodes/` and `edges/`. These two aggregate files are the source of the graph used in this study.

Alternatively, PowerShell 7 can download the same files directly with the following commands. Use a new checkout or an empty destination so existing source files are preserved.

```powershell
$target = 'data/cache/optimuskg/doi_10_7910_DVN_IYNGEV/2.0/recovery/aggregate'
New-Item -ItemType Directory -Path $target -Force | Out-Null
if (Test-Path -LiteralPath "$target/nodes.parquet") { throw 'nodes.parquet already exists; keep your existing file.' }
if (Test-Path -LiteralPath "$target/edges.parquet") { throw 'edges.parquet already exists; keep your existing file.' }
Invoke-WebRequest 'https://dataverse.harvard.edu/api/access/datafile/13835035' -OutFile "$target/nodes.parquet"
Invoke-WebRequest 'https://dataverse.harvard.edu/api/access/datafile/13835018' -OutFile "$target/edges.parquet"
```

The [official release metadata](https://dataverse.harvard.edu/api/datasets/:persistentId/versions/2.0?persistentId=doi:10.7910/DVN/IYNGEV) identifies file 13835035 as `nodes.parquet` (163,857,672 bytes) and file 13835018 as `edges.parquet` (177,526,798 bytes). `check_data.py` checks both files against the study's SHA-256 values.

### DrugBank

Use your own authorized account on the [DrugBank downloads page](https://go.drugbank.com/releases/latest). Select the **full database XML for All drugs**, extract the archive, and save the full XML as `data/drugbank_all_full_database.xml`. Save the linked **Schema Definition** from that release as `data/drugbank.xsd`. The parser requires the XML root attribute `version="5.1"` and the DrugBank namespace; other schema versions stop preparation. The CSV and SDF exports are different inputs and cannot replace the full XML.

On 26 November 2026, the official page reported that academic downloads were temporarily paused. Use an existing authorized full export or obtain access through DrugBank when the XML download is unavailable.

For command-line access, use the **Command / Example** supplied by DrugBank for the release and account you are authorized to use, then extract and name the two files as above. The provider controls account access and download availability; this repository does not supply or redistribute DrugBank files.

The study's local XML records `version="5.1"` and `exported-on="2026-04-23"`. This header does not identify a patch release. Preparation checks the supported XML/XSD structure and records the supplied file hashes. Retain the provider's release information and checksum: a different export can change structured drug mappings or features even when its header says 5.1.

### Check placement

```sh
python check_data.py
```

This standard-library command reads the files without changing them. It reports missing or misnamed files, verifies both OptimusKG checksums, and checks the DrugBank XML and XSD root elements. A successful placement check confirms the raw inputs; it does not create the training corpus or validate every DrugBank record.

All of `data/` is excluded from Git. Keep the source files unchanged and retain DrugBank-derived artifacts locally under the applicable data terms.

## Supported preparation routes

### Prepare from the raw databases

Install a suitable PyTorch build, then the full preparation dependencies:

```sh
python -m pip install -r requirements-preprocess.txt
python main.py
```

If the configured corpus is absent, this command runs canonicalization, R0F1 split reconstruction, disease-gene leakage cleaning, SideInfo and frozen D1 feature construction, sampling-plan reconstruction, and portable-corpus validation before training. It then completes the three-cycle FULL plan, development evaluation, rankings and path explanations, and writes `outputs/<run>/report.html`. All database parsing and encoder inference run locally. No previous research directory, trained checkpoint or historical execution receipt is needed.

The [canonical preparation specification](raw_preparation.md) describes entity matching, fact and evidence handling, verified source exclusions, and reconstruction of the original CSR traversal order.

For preparation without optimizer updates:

```sh
python main.py --mode prepare
```

The raw recipe supports `cell: R0F1`. It reconstructs the historical candidate exclusions, final-holdout reservations, five-fold memberships and independent support/contra inner reservations. Membership and mask digests must match the recorded R0F1 recipe. The three global cycles and all conditional U records are also checked. Different memberships or another cell stop preparation; the program does not substitute a new random split. Historical checkpoint and corpus hashes in the conditional seed recipe are fixed seed inputs; their files are not loaded.

### Configure paths, devices and downloads

Copy [configs/raw.yaml](../configs/raw.yaml) when changing source or cache locations. It contains these fields:

| Field | Purpose |
| --- | --- |
| `nodes`, `edges`, `drugbank_xml`, `drugbank_xsd` | Read-only local source files. |
| `stage_root` | Derived stages and their receipts; default `data/interim/raw_R0F1`. |
| `open_targets_cache` | Pinned Open Targets 25.06 datasource Parquet cache. |
| `model_cache` | Pinned ChemBERTa and BioBERT files; default `data/cache/pretrained`. |
| `cell` | `R0F1`. |
| `feature_device` | `cuda` or `cpu` for frozen encoder inference; default `cuda`. |
| `feature_batch_size` | Positive encoder batch size; default 16. |

Pass the copy explicitly:

```sh
python main.py --mode prepare --raw-config /path/to/raw.yaml
```

The training configuration passed with `--config` determines `manifest`, `output_dir`, training `device` and `precision`. Its `manifest` must end with `manifest.json`. The default is `configs/default.yaml`; `configs/cpu.yaml` selects CPU/FP32 training. To prepare features on CPU as well, set `feature_device: cpu` in the raw configuration. Relative paths inside either YAML resolve from the repository root.

The separate `--report-config` option defaults to `configs/report.yaml`. Its `explanation_queries: 10`, `paths_per_query: 5` and `ranking_top_k: 10` settings limit report output after evaluation. They do not affect the prepared corpus, training plan or development candidate membership. See [report outputs](reproducibility.md#run-completion-and-outputs).

Missing public auxiliary resources are downloaded automatically and checked by size and SHA-256 before use:

| Resource | Pinned input and cache |
| --- | --- |
| Open Targets | Release 25.06 `association_by_datasource_indirect`: 20 Parquet files, 99,508,052 bytes total. [The source recipe](../configs/preprocessing/opentargets.json) fixes file identities and the datasource exclusion policy. |
| ChemBERTa and BioBERT | The revisions listed under [Entity features](#entity-features), with pinned weights, tokenizer and configuration files. The two weight files total 612,098,256 bytes (about 612 MB); tokenizer/configuration files add a small amount. |

Preparation uses Open Targets associations from the allowed sources to replace the original disease-gene evidence. It does not treat association presence or a source score as an explicit support/refutation label. Encoder downloads supply public model files; DrugBank XML, SMILES and other local records are not uploaded.

To use only verified caches:

```sh
python main.py --mode prepare --raw-config /path/to/raw.yaml --offline
```

Populate the caches during an online preparation or transfer the exact pinned cache files under their original filenames. Offline mode reports missing resources and stops. Existing files with unexpected hashes are preserved and rejected.

### Resume preparation or rebuild changed inputs

Rerun the same command after an interruption. Each completed stage has a receipt and a file-hash inventory under `stage_root`; reuse requires matching files and the same build identity. The identity includes source hashes, code and recipe hashes, dependency versions, the feature device and batch size. The feature stage can reuse its completed output types. An unfinished stage without a completed receipt runs in a new attempt directory, except for the feature stage's supported recovery. Failed attempt directories remain available for inspection.

Preparation writes its events, error tracebacks and completion report under `output_dir/preparation/<attempt-id>/`. If a stale `.prepare.lock` remains after a process crash, check the recorded PID and confirm that preparation has stopped before removing that lock.

When source files, code, bound packages or feature settings change, choose a new `stage_root` and a new final corpus directory through the training YAML's `manifest` field. Keep the old artifacts for comparison. If the final `manifest.json` already exists, `--mode prepare` requires the raw source files and a matching source, recipe and build identity before reuse. It rejects a mismatch and never overwrites the corpus. Use `--mode validate` to check an existing corpus without raw sources or preprocessing dependencies. A normal training invocation with an existing corpus skips raw preparation and validates the corpus itself.

The final corpus is published only after its complete validation succeeds; an interrupted publication leaves an `.attempt_...` directory. Existing final output directories are never overwritten.

### Use an existing or independently prepared corpus

If a complete portable corpus is already available locally, keep its files and subdirectories together under `data/processed/corpus/`, so that the manifest is exactly `data/processed/corpus/manifest.json`. Then run `python main.py --mode validate`, followed by `python main.py`.

If you have a completed original research workspace, export one cell explicitly:

```text
python export_data.py --research-root /path/to/research --cell R0F1 --d1-root /path/to/frozen/features --output data/processed/corpus
```

Export requires a completed masked corpus and a sealed sampling plan. It verifies the sampling seal, source hashes, membership digests, split mask, canonical graph, SideInfo arrays and D1 input provenance. The new directory contains training and both development panels, with the original 3G group order and conditional records. For a relocated research workspace, supply the former root from the manifests via `--recorded-root`.

For normalized inputs prepared independently, copy `schemas/source.example.json`, fill its actual paths, counts, semantic audit declarations and hashes, then run:

```text
python prepare_data.py --source /path/to/source.json --output data/processed/corpus
python main.py
```

`prepare_data.py` copies and validates the source files, preserving labels, missing values, splits and training samples. Source paths resolve beside the source-map JSON. The default output is `data/processed/corpus`. Choose a new directory if the output already exists. Failures point to a traceback log under `outputs/logs`.

`evi_hgt.data.raw_features` provides `fingerprint`, `profile`, `pack_drug` and `pack_disease` for raw feature vectors. The raw workflow invokes these helpers and requires RDKit from `requirements-preprocess.txt`. Loading an existing portable corpus uses its prepared arrays and needs `requirements.txt`.

## Source map and manifest

A source map has three keys:

```json
{
  "schema": "EVIHGT_SOURCE_MAP_V1",
  "manifest": {"schema": "EVIHGT_PORTABLE_CORPUS_V1"},
  "sources": {
    "nodes.parquet": {"path": "../licensed_inputs/nodes.parquet", "sha256": "actual SHA-256"}
  }
}
```

The `manifest` object must contain the complete fields in the example. `sources` must cover every referenced file. The preparer fills `manifest.files`, an exact mapping of relative POSIX paths to SHA-256 hashes. Portable paths must remain within the corpus directory; absolute paths, parent traversal and escaping symlinks are rejected. Every input is hashed on load. The manifest itself determines the corpus identity used by training/checkpoints.

| Manifest field | Required content |
| --- | --- |
| `training_seed` | Integer from 0 through 2^31 - 1. |
| `token_buckets` | 8192, preserving the implemented token namespace. |
| `nodes`, `edge_catalog` | Canonical node Parquet and ordered relation catalog JSON. |
| `fact_files`, `evidence_files` | Ordered lists of `{path, component, rows}`. Rows from different files use a common schema. |
| `csr` | Relative paths for `indptr`, `indices`, `fact_ids` NumPy arrays. |
| `target_map`, `forbidden_facts` | Target-equivalent canonical facts and permanently removed split facts. |
| `training_pairs`, `observed_pairs`, `blocked_pairs`, `eligibility` | Verified clinical positives, role-free U exclusions and candidate domains. |
| `panels` | `validation` and `auxiliary_validation` paths; optional `train_probe` is read but not automatically scored. |
| `raw_features`, `pretrained` | Typed SideInfo and D1 array paths described below. |
| `schedule` | `G`, `groups`, `conditional_records` and an auditable `sampling_provenance`. |
| `semantics` | Explicit upstream audit declarations from the example; `unknown_is_negative` must be false. |

Base semantic declarations on source audits and record unresolved fields as such. Use verified structured relations for clinical labels. DrugBank `known-action=no` alone does not qualify as counter evidence or a negative drug-disease label.

## Canonical graph and evidence

`nodes.parquet` requires `global_node_id: integer`, `node_id: string`, and `node_type: string`. Global IDs are unique, contiguous and zero based; canonical string IDs are unique. The implemented type order is:

```text
ANA, BPO, CCO, DIS, DRG, EXP, GEN, MFN, PHE, PWY
```

The ordered edge catalog is a JSON array of objects with `source_node_type`, `relation` and `target_node_type`. The same canonical edge type is used for either legal traversal direction; independent path direction is represented separately.

Canonical fact tables require:

| Column | Meaning |
| --- | --- |
| `canonical_fact_id` | Unique nonnegative integer fact identity. |
| `source_global_id`, `target_global_id` | Canonical node integer IDs. |
| `source_node_id`, `target_node_id` | Corresponding canonical string IDs. |
| `source_node_type`, `target_node_type` | Verified endpoint types. |
| `relation` | Verified canonical relation; no guessed aliases. |
| `undirected` | Boolean declaring whether reverse traversal is legal. |

Evidence tables require `evidence_record_id` (globally unique nonnegative integer), `canonical_fact_id`, `properties` (JSON object string), `raw_direct_sources` and `raw_indirect_sources` (string lists), `source_row_number`, `source_component_path`, `overlap_category`, `edge_support_observed` and `edge_counter_evidence_observed` (Booleans). The loader takes component names from the file descriptors and matches evidence by component. Multiple provenance tokens can describe one record.

`overlap_category` is null or one of `EXACT_DUPLICATE`, `PROBABLE_DUPLICATE`, `SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE`, `UNRESOLVED_OVERLAP`. The numeric property fields are `expression_rank`, `evidence_score`, `evidence_count`, `evidence_index`, `number_of_pmids`, `highest_clinical_trial_phase`; `qualifier_not` is the Boolean property field. Missing values have false masks. Other property/evidence tokens and provenance retain the exact implemented hashing and missing-token rules. Direct evidence logic is available only for verified ANA-to-GEN `EXPRESSION_PRESENT`/`EXPRESSION_ABSENT` facts in the implemented model.

CSR arrays are one-dimensional integer arrays. `indptr` has node-count + 1 elements; `indices` and `fact_ids` have one element per physical traversal. Preserve the original physical traversal order: it affects bounded neighbor admission. Query extraction uses two hops, at most 1000 nodes and 5000 messages. Path candidates use lengths 2-4, at most 16 paths, beam width 32 and at most 64 neighbors per expansion. No entire graph is copied to the GPU.

`target_map` has `drug_id`, `disease_id`, `canonical_fact_id`. It must include every direct or reverse drug-disease fact regardless of relation name, plus all verified semantically equivalent facts. `forbidden_facts` has `canonical_fact_id`. Held-out target facts and their evidence must already be physically absent from canonical tables and CSR; the loader rejects survivors. Each query additionally removes its own target-equivalent fact IDs. The canonical expression-message policy then preserves the forward ANA-to-GEN expression message when the selected traversals contain only its reverse; independent path direction remains unchanged.

## Clinical roles and exclusions

`training_pairs` has `drug_id`, `disease_id`, `channel`, `raw_relation`. Support positives require `channel=support`, `raw_relation=INDICATION`; contra positives require `channel=unsuitability`, `raw_relation=CONTRAINDICATION`. The upstream indication inclusion policy must already be verified. The two positive channels are independent and may coexist for a pair. Pairs without an observed label remain unlabeled.

`observed_pairs` and `blocked_pairs` contain unique `drug_id`, `disease_id` pairs only. Observed clinical pairs are excluded from all U draws. Blocked pairs include development/holdout exclusions and cannot enter training. `eligibility` is JSON with string lists `eligible_drugs` and `eligible_diseases`; legal U lies within their Cartesian product after exclusions.

`validation` contains `drug_id`, `disease_id`, `role=P|U` and `raw_relation` (`INDICATION` for P, null for U). `auxiliary_validation` contains only verified contraindication P rows, with `raw_relation=CONTRAINDICATION`. Contra evaluation uses these positives and the same validation U pool. U denotes unobserved candidate pairs. Include all development members in the graph and training exclusions.

## Entity features

`raw_features` maps `drug_ids`, `drug_values`, `disease_ids`, `disease_values` to NumPy files. IDs follow ascending canonical global ID within type. Values are `uint8` binary arrays:

- Drug: 2048 Morgan radius-2 bits, 512 hashed biology-profile bits, two explicit availability bits (2562 dimensions).
- Disease: 512 hashed ontology bits, 512 hashed phenotype bits, two availability bits (1026 dimensions).
- Query nodes receive the corresponding typed block in a 3588-dimensional vector; other node types have zeros. An unavailable family must contain zero values.

The original profile namespace/version and hashing are retained in the helper code. Drug biology may use verified drug-gene/DrugBank structured drug-gene sources; disease profiles use ontology ancestry and HPO phenotypes. Target labels, target-containing clinical text and supervised prediction features are forbidden.

`pretrained` maps `drug_ids`, `drug_values`, `drug_available`, `disease_ids`, `disease_values`, `disease_available`, `metadata` to files. Values are frozen finite `float32 [entity_count,768]`; availability is Boolean. Missing vectors are exactly zero with false availability. See `schemas/d1_metadata.example.json` for required declarations. The pinned encoders are ChemBERTa `seyonec/ChemBERTa-zinc-base-v1` revision `761d6a18cf99db371e0b43baf3e2d21b3e865a20` (max length 512), and BioBERT `dmis-lab/biobert-v1.1` revision `551ca18efd7f052c8dfa0b01c94c2a8e68bc5488` (max length 128). Pool the final hidden state over non-padding, non-special tokens. Drug inputs are verified canonical/structured SMILES; disease inputs are canonical names. No fine-tuning is performed by the model.

The target-masking checks cover the constructed input graphs. Whether the encoders' original pretraining corpora contain the drug-disease associations under evaluation has not been established; the recorded status is `NOT_VERIFIED`. See the [model card](model_card.md#leakage-and-provenance-boundaries) for the scope of these checks.

## Exact three-cycle plan

`schedule.groups` is Parquet with integer `step`, `cycle`, `position` and string `drug_id`, `disease_id`, `role`. Roles are `support`, `unsuitability`, `U`. Steps are 1 through 3G; positions are contiguous and zero based within each step. Every group has at least one support P, at least one contra P and exactly one shared global U. Every complete cycle covers every positive exactly once; G equals the smaller positive-channel count. The first G steps use global U. The remaining 2G use the original conditional records. The loader uses the supplied sample order and U draws.

`schedule.conditional_records` is a JSON list in step order. Each entry contains:

```json
{
  "logical_step": 2,
  "group_index": 0,
  "support_positive_pairs": [["drug", "disease"]],
  "contra_positive_pairs": [["other_drug", "other_disease"]],
  "original_global_u": ["unobserved_drug", "disease"],
  "original_support_u_count": 1,
  "disease_unlabeled_pairs": {"disease": [["local_drug", "disease"]]},
  "disease_status": {"disease": "DISEASE_CONDITIONAL_UNIFORM"},
  "draw_seed_by_disease": {"disease": 123},
  "extra_pairs": [["local_drug", "disease"]]
}
```

The example assumes `G=1`. Each support-positive disease has one legal local U. If none exists, use an empty list and `NO_LEGAL_U_ORIGINAL_PAIRING_FALLBACK`; retain the original global U comparison. `extra_pairs` lists distinct local U not already present in the original group, in first encounter order. Contra comparisons continue to use the original global U. Retain the plan provenance and draw seeds. Assign a new run identity to a regenerated plan.

## Software-only validation

Run `python main.py --smoke` for a CPU check of the complete main workflow without databases or network access. It creates a tiny corpus marked `SYNTHETIC_TEST_ONLY`, performs three optimizer updates and writes evaluation results, rankings, explanations and a report. The fixture exercises loading, target masking, evidence tensorization, query and path extraction, feature validation, and the 3G schedule. It does not test reconstruction of the four original database files. The real raw-preparation verification is recorded separately in the [reproducibility guide](reproducibility.md#software-verification).
