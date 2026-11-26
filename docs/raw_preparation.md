The canonical preparation stage reads the two OptimusKG 2.0 aggregate Parquet files and a local DrugBank XML/XSD pair. It produces the node index, canonical facts, source evidence, DDA supervision, target-fact map, and CSR graph used by subsequent preparation stages. The raw databases remain unchanged. DrugBank files and derived records stay on the local machine and are excluded from the repository.

`build_canonical` verifies the pinned OptimusKG 2.0 file sizes, MD5 values, and SHA-256 values. Other aggregate releases fail this identity check. For DrugBank, it compiles the supplied XSD, checks the XML root namespace and version `5.1`, and streams the structured fields into Parquet. The audit records the actual XML/XSD hashes and export header. It explicitly records that full-document XSD validation was not performed; the header does not identify an unreported DrugBank patch release.

The builder preserves aggregate JSON properties, including fields it does not interpret. It derives ten node partitions and 27 edge partitions, with an aggregate row index retained in each partition. The aggregate drug and disease properties supply the fields needed by later feature preparation, so a second download of typed node files is unnecessary.

Canonical entity IDs are assigned by sorted node type and entity identifier, with a global integer ID and an integer ID within each type. DrugBank drugs merge into existing entities only through a unique supported ChEMBL mapping; unmatched or ambiguous records receive local DrugBank identifiers. Structured partner records map to genes through unique UniProt identifiers present in the aggregate gene properties.

A canonical fact key includes endpoint types and identifiers, relation, and direction. Same-type undirected endpoints use a canonical orientation. Repeated keys produce one fact, while each source observation remains an evidence row. Overlapping observations retain `UNRESOLVED_OVERLAP` and `independent_vote_eligible=False`; a source name or repeated reference does not establish independent evidence. A DrugBank partner observation that matches an existing drug-gene fact reuses its fact ID. New supported partner relations receive new fact IDs.

The pinned aggregate contains 142 `DIS-DIS` rows whose source actually resolves to `PHE`. The builder records their source row indices, original fields, parent hash, and exclusion reason, then excludes those rows from canonicalization. It rejects endpoint anomalies outside this verified pattern. Raw aggregate rows are never removed or rewritten.

DrugBank drug-interaction pairs and normalized descriptions are compared for overlap. Without verified shared-release alignment, a matching pair and description is `PROBABLE_DUPLICATE`. These interaction records do not add synergy labels or DDA labels. DrugBank indication text and `known-action=no` do not create negative or refuting DDA supervision.

DDA policy B selects an `INDICATION` with direct `DRUG_CENTRAL` provenance or highest clinical-trial phase at least 2. The second observed channel comes from `CONTRAINDICATION`. `OFF_LABEL_USE` remains context. Unobserved channel values are null and their masks are false. Anatomy-gene `EXPRESSION_ABSENT` describes a different proposition from clinical contraindication and remains a separate relation.

The artifact tree contains these paths:

| Path | Content |
| --- | --- |
| `nodes/node_index.parquet` | Canonical entity identifiers and integer IDs |
| `partitions/nodes/*.parquet` | Lossless aggregate node partitions |
| `canonical_facts/*.parquet` | Canonical facts, including new DrugBank partner facts |
| `evidence_records/*.parquet` | Source observations and conservative overlap decisions |
| `drugbank/*.parquet` | Drug mappings, supplemental nodes, partner membership, and unresolved partners |
| `supervision/dda_supervision.parquet` | Nullable DDA targets, masks, and legacy split annotations |
| `views/query_target_fact_map.parquet` | Clinical target pairs mapped to canonical fact IDs |
| `graph/csr/` | Memory-mappable graph arrays and relation catalog |
| `manifest/` | Input identities, partition metadata, exclusions, inventory, and completion record |

The full canonical CSR still contains clinical facts. Later preparation applies the requested experiment cell's physical exclusions and constructs its training and evaluation inputs. The legacy split annotations in the canonical output are not a replacement for those experiment definitions.

An existing destination is never overwritten. Each run creates fresh staging directories, writes audits outside the data tree, checks source identities again before publication, and atomically renames the completed graph into place. Interrupted staging directories remain available for diagnosis. The completion manifest records the scientific checks that actually ran; it does not assert that a historical authorization or a separate test suite passed.

The full reconstruction checked on 26 November 2026 produced the following counts with the local DrugBank export used by the research project:

| Artifact | Count |
| --- | ---: |
| Raw aggregate nodes | 190,531 |
| Raw aggregate edges | 21,813,816 |
| Excluded endpoint anomalies | 142 |
| Final nodes | 202,131 |
| Canonical facts | 21,829,430 |
| Evidence rows | 21,847,387 |
| CSR traversals | 40,997,452 |
| Policy B support pairs | 43,973 |
| Retained contraindication pairs | 11,718 |

The 73 compared Parquet tables had matching schemas and all ordered values, including the canonical fact IDs, evidence IDs, DrugBank mappings, supervision, node partitions, and target-fact map. All four CSR arrays (`indptr`, `indices`, `fact_ids`, and `edge_type_ids`) had the same complete SHA-256 values as the research artifacts. The supervision statistics also matched.

CSR construction uses the explicit `historical-relation-precedence-v1` order in `csr_order_recipe.py`. This small recipe contains partition names and typed relation names. It was recovered from relation-group precedence in every historical CSR row: consecutive groups supply ordering constraints, and a deterministic topological ordering satisfies them. The resulting arrays reproduce the original physical traversal order. This is necessary because capped query extraction admits neighbors in their encounter order; graph-edge equivalence alone does not establish equivalent capped queries. Relation tuples absent from the recipe follow the listed tuples in lexical order.

With the final builder, 256 deterministically selected target pairs produced identical node sets, ordered fact IDs, and ordered traversals under limits of two hops, 1,000 nodes, and 5,000 edges, with each query's target facts hidden. This acceptance check covers bounded extraction from the complete canonical CSR. It does not evaluate a trained model or substitute for validating the final experiment cell's physical mask and query assembly. A different DrugBank export may change mapped entities and facts, so the observed counts and historical byte equality apply to the checked raw input identities.
