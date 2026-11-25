# Data preparation and portable input contract

The trainer reads `data/processed/corpus/manifest.json`. Its corpus contains canonical entity, fact and evidence tables, a masked CSR graph, fixed entity features, verified clinical memberships and a fixed three-cycle sampling plan. The public preparation tools import those prepared artifacts. The raw-to-training preprocessing entry point is not yet published.

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

Use your own authorized account on the [DrugBank downloads page](https://go.drugbank.com/releases/latest). Select the **full database XML for All drugs**, extract the archive, and save the full XML as `data/drugbank_all_full_database.xml`. Save the linked **Schema Definition** as `data/drugbank.xsd`. The CSV and SDF exports are different inputs and cannot replace the full XML.

For command-line access, use the **Command / Example** supplied by DrugBank for the release and account you are authorized to use, then extract and name the two files as above. DrugBank's academic downloads were temporarily paused when checked on **25 November 2026**. The provider controls account access and download availability; this repository does not supply or redistribute DrugBank files.

The study's local XML records `version="5.1"` and `exported-on="2026-04-23"`. This header does not identify a patch release. Retain the release information and checksum supplied with your authorized download when preparing a new corpus.

### Check placement

```sh
python check_data.py
```

This standard-library command reads the files without changing them. It reports missing or misnamed files, verifies both OptimusKG checksums, and checks the DrugBank XML and XSD root elements. A successful placement check confirms the raw inputs; it does not create the training corpus or validate every DrugBank record.

All of `data/` is excluded from Git. Keep the source files unchanged and retain DrugBank-derived artifacts locally under the applicable data terms.

## Supported preparation routes

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

`evi_hgt.data.raw_features` provides `fingerprint`, `profile`, `pack_drug` and `pack_disease` to construct raw feature vectors. Install the optional `rdkit` package to call `fingerprint`. The other workflows use prepared arrays.

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

Tests generate a small corpus marked `SYNTHETIC_TEST_ONLY` in temporary output directories. The fixture exercises loading, target masking, evidence tensorization, query and path extraction, feature validation, and the 3G schedule.
