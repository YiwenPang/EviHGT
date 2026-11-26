"""Raw-input canonicalization regression tests; all records are synthetic."""
from importlib.util import find_spec


def test_raw_canonical_public_entrypoint_exists():
    assert find_spec("evi_hgt.preprocessing.canonical") is not None, (
        "The public repository must build its canonical graph from raw inputs"
    )


def test_csr_replays_historical_relation_order_for_capped_queries(tmp_path):
    from evi_hgt.data.native_csr import extract_csr_subgraph_native
    from evi_hgt.preprocessing.raw_source.graph_builder import build_csr_index
    facts = pl.DataFrame({
        "canonical_fact_id": [0, 1, 2], "source_global_id": [0, 0, 0],
        "target_global_id": [1, 2, 3], "source_node_type": ["DRG"] * 3,
        "target_node_type": ["DIS"] * 3,
        "relation": ["INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE"],
        "undirected": [False] * 3,
    })
    partition = tmp_path / "drug_disease.parquet"
    facts.write_parquet(partition)
    for attempt in range(8):
        output = tmp_path / f"csr_{attempt}"
        build_csr_index([partition], node_count=4, output_dir=output)
        query = extract_csr_subgraph_native(output, seed_node_ids=[0],
            max_hops=1, max_nodes=4, max_edges=1)
        assert query.fact_ids == (2,)
        assert query.node_ids == (0, 3)


def test_public_entrypoint_refuses_existing_destination(tmp_path):
    from evi_hgt.preprocessing.canonical import build_canonical
    destination = tmp_path / "processed"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_canonical("missing", "missing", "missing", "missing", destination,
                        audit_dir=tmp_path / "outputs")
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_public_entrypoint_rejects_unknown_release_before_conversion(tmp_path):
    from evi_hgt.preprocessing.canonical import build_canonical
    from evi_hgt.preprocessing.raw_source.phase3_recovery_build import FormalRecoveryError
    sources = [tmp_path / name for name in ("nodes.parquet", "edges.parquet", "drugbank.xml", "drugbank.xsd")]
    for path in sources:
        path.write_bytes(b"synthetic wrong release")
    destination = tmp_path / "data/processed/canonical"
    with pytest.raises(FormalRecoveryError, match="aggregate binding"):
        build_canonical(*sources, destination, audit_dir=tmp_path / "outputs")
    assert not destination.exists()
    assert not (destination.parent / ".staging").exists()


def test_invalid_completion_cannot_publish_staging(tmp_path):
    from evi_hgt.preprocessing.raw_source.phase3_recovery_build import FormalRecoveryError, publish_formal_staging
    staging = tmp_path / "data/processed/.staging/build"
    staging.mkdir(parents=True)
    final = tmp_path / "data/processed/canonical"
    with pytest.raises(FormalRecoveryError, match="completion status"):
        publish_formal_staging(staging, final, {})
    assert staging.exists() and not final.exists()


def test_drugbank_envelope_records_limited_validation_scope(tmp_path):
    from evi_hgt.preprocessing.canonical import _drugbank_envelope
    xml = _write_fixture(tmp_path / "drugbank.xml")
    xsd = tmp_path / "drugbank.xsd"
    xsd.write_text('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" '
        'targetNamespace="http://www.drugbank.ca"><xs:element name="drugbank" '
        'type="xs:anyType"/></xs:schema>', encoding="utf-8")
    envelope = _drugbank_envelope(xml, xsd)
    assert envelope["xml_header"]["version"] == "5.1"
    assert envelope["full_document_xsd_validation_performed"] is False


def test_raw_aggregate_to_canonical_graph_with_drugbank(tmp_path):
    import hashlib
    import json
    from evi_hgt.preprocessing.canonical import _build_config
    from evi_hgt.preprocessing.raw_source.phase3_formal_pipeline import run_authorized_phase3_recovery_build
    raw = tmp_path / "raw"
    raw.mkdir()
    nodes = raw / "nodes.parquet"
    edges = raw / "edges.parquet"
    node_rows = [("CHEMBL1", "DRG"), ("CHEMBL2", "DRG"), ("DIS:1", "DIS"),
                 ("DIS:2", "DIS"), ("DIS:3", "DIS"), ("GEN:1", "GEN"), ("PHE:1", "PHE")]
    pl.DataFrame([{"id": identifier, "label": kind, "properties": json.dumps(
        {"name": identifier, "preserved_unknown_field": 17, "associated_proteins":
         [{"id": "P12345", "source": "uniprot_swissprot"}] if kind == "GEN" else []})}
        for identifier, kind in node_rows]).write_parquet(nodes)
    rows = [("CHEMBL1", "DIS:1", "DRG-DIS", "INDICATION", True),
            ("CHEMBL2", "DIS:2", "DRG-DIS", "CONTRAINDICATION", True),
            ("CHEMBL1", "DIS:3", "DRG-DIS", "OFF_LABEL_USE", True),
            ("CHEMBL1", "GEN:1", "DRG-GEN", "TARGET", False),
            ("CHEMBL1", "GEN:1", "DRG-GEN", "TARGET", False),
            ("CHEMBL1", "CHEMBL2", "DRG-DRG", "SYNERGISTIC_INTERACTION", True),
            ("DIS:1", "DIS:2", "DIS-DIS", "IS_A", True),
            ("PHE:1", "DIS:2", "DIS-DIS", "IS_A", True)]
    pl.DataFrame([{"from": a, "to": b, "label": family, "relation": rel,
        "undirected": undirected, "properties": json.dumps({
            "sources": {"direct": ["DRUG_CENTRAL"], "indirect": []},
            "highest_clinical_trial_phase": 3.0 if rel == "INDICATION" else None,
            "reference_ids": [], "interaction_description": "synthetic"})}
        for a, b, family, rel, undirected in rows]).write_parquet(edges)
    xml = _write_fixture(raw / "drugbank.xml")
    raw_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (nodes, edges, xml)}
    interim = tmp_path / "data/interim/drugbank"
    convert_drugbank_xml(xml, interim)
    identities = {kind: {"size_bytes": path.stat().st_size,
        "md5": hashlib.md5(path.read_bytes()).hexdigest(),
        "sha256": raw_hashes[path]} for kind, path in (("nodes", nodes), ("edges", edges))}
    destination = tmp_path / "data/processed/canonical"
    audit = tmp_path / "outputs/canonical_audit"
    config = _build_config(nodes, edges, destination, audit,
        tmp_path / "data/processed/.staging", interim, identities, expected_exclusions=1)
    result = run_authorized_phase3_recovery_build(config=config, project_root=tmp_path,
        enforce_official_baselines=False)
    assert result.processed_root == destination
    assert (result.node_count, result.canonical_fact_count, result.evidence_record_count,
            result.traversal_edge_count) == (8, 6, 8, 11)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
               for path, digest in raw_hashes.items())
    facts = pl.read_parquet(destination / "canonical_facts/drug_gene.parquet")
    evidence = pl.read_parquet(destination / "evidence_records/drug_gene.parquet")
    db_evidence = pl.read_parquet(destination / "evidence_records/drugbank_drug_gene.parquet")
    assert facts.height == 1 and evidence.height == 2 and db_evidence.height == 1
    assert db_evidence["canonical_fact_id"].to_list() == facts["canonical_fact_id"].to_list()
    assert not evidence["independent_vote_eligible"].any()
    supervision = pl.read_parquet(destination / "supervision/dda_supervision.parquet")
    assert supervision.filter(pl.col("raw_relation") == "CONTRAINDICATION")["support_target"].null_count() == 1
    assert supervision.filter(pl.col("raw_relation") == "OFF_LABEL_USE")["support_mask"].to_list() == [False]
    assert pl.read_parquet(destination / "views/query_target_fact_map.parquet").height == 3
    completion = json.loads((destination / "manifest/completion.json").read_text(encoding="utf-8"))
    assert completion["all_completion_gates_passed"] is True
    assert completion["release_baselines_verified"] is False
    assert completion["canonical_exclusion"]["row_count"] == 1
    assert "prepublication_pytest_passed" not in completion["completion_gates"]
    with pytest.raises(FileExistsError):
        run_authorized_phase3_recovery_build(config=config, project_root=tmp_path,
            enforce_official_baselines=False)


from pathlib import Path

import polars as pl

from evi_hgt.preprocessing.raw_source.drugbank_loader import (
    convert_drugbank_xml,
    iter_drugbank_records,
)

DRUGBANK_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca" version="5.1" exported-on="2026-04-23">
  <drug type="small molecule" created="2005-06-13" updated="2026-01-01">
    <drugbank-id primary="true">DB00001</drugbank-id>
    <drugbank-id>BIOD00001</drugbank-id>
    <name>Fixture Drug</name>
    <groups><group>approved</group><group>investigational</group></groups>
    <atc-codes>
      <atc-code code="A01AA01">
        <level code="A01AA">Fixture subgroup</level>
        <level code="A01A">Fixture group</level>
      </atc-code>
    </atc-codes>
    <external-identifiers>
      <external-identifier><resource>ChEMBL</resource><identifier>CHEMBL1</identifier></external-identifier>
    </external-identifiers>
    <nested-reference><drug><drugbank-id primary="true">DB99999</drugbank-id><name>Not a top-level record</name></drug></nested-reference>
    <drug-interactions>
      <drug-interaction><drugbank-id>DB00002</drugbank-id><name>Second Drug</name><description>Structured interaction text.</description></drug-interaction>
    </drug-interactions>
    <targets>
      <target>
        <id>BE0000001</id><name>Fixture target</name><organism>Humans</organism>
        <actions><action>inhibitor</action></actions>
        <references><articles><article><ref-id>A1</ref-id><pubmed-id>12345</pubmed-id><citation>Fixture</citation></article></articles><textbooks/><links/><attachments/></references>
        <known-action>no</known-action>
        <polypeptide id="P12345" source="Swiss-Prot">
          <name>Target protein</name><general-function/><specific-function/><gene-name>GENE1</gene-name><locus/><cellular-location/><transmembrane-regions/><signal-regions/><theoretical-pi/><molecular-weight/>
          <chromosome-location/><organism ncbi-taxonomy-id="9606">Humans</organism>
          <external-identifiers><external-identifier><resource>UniProtKB</resource><identifier>P12345</identifier></external-identifier></external-identifiers>
          <synonyms/><amino-acid-sequence format="FASTA"/><gene-sequence format="FASTA"/><pfams/><go-classifiers/>
        </polypeptide>
      </target>
    </targets>
    <enzymes/><carriers/><transporters/>
  </drug>
  <drug type="biotech" created="2005-06-13" updated="2026-01-01">
    <drugbank-id primary="true">DB00002</drugbank-id>
    <name>Second Drug</name><groups><group>experimental</group></groups>
    <atc-codes/><external-identifiers/><targets/><enzymes/><carriers/><transporters/>
  </drug>
</drugbank>
"""

def _write_fixture(path: Path) -> Path:
    path.write_text(DRUGBANK_FIXTURE, encoding="utf-8")
    return path

def test_streaming_parser_extracts_only_structured_fields(tmp_path: Path) -> None:
    xml_path = _write_fixture(tmp_path / "drugbank.xml")

    records = list(iter_drugbank_records(xml_path))

    assert [record.primary_drugbank_id for record in records] == ["DB00001", "DB00002"]
    first = records[0]
    assert first.name == "Fixture Drug"
    assert first.drug_type == "small molecule"
    assert first.groups == ("approved", "investigational")
    assert first.atc_codes[0].code == "A01AA01"
    assert first.external_identifiers[0].resource == "ChEMBL"
    partner = first.partners[0]
    assert partner.category == "target"
    assert partner.actions == ("inhibitor",)
    assert partner.known_action == "no"
    assert partner.gene_name == "GENE1"
    assert partner.uniprot_ids == ("P12345",)
    assert partner.reference_ids == ("PUBMED:12345", "DRUGBANK_REF:A1")
    assert first.drug_interactions[0].drugbank_id == "DB00002"
    assert first.drug_interactions[0].description == "Structured interaction text."
    assert not hasattr(first, "dda_label")

def test_conversion_writes_columnar_interim_tables(tmp_path: Path) -> None:
    xml_path = _write_fixture(tmp_path / "drugbank.xml")
    output_dir = tmp_path / "interim"

    summary = convert_drugbank_xml(xml_path, output_dir, batch_size=1)

    assert summary.drug_count == 2
    assert summary.partner_row_count == 1
    assert summary.output_dir == output_dir.resolve()
    drugs = pl.read_parquet(output_dir / "drugbank_drugs.parquet")
    partners = pl.read_parquet(output_dir / "drugbank_partners.parquet")
    interactions = pl.read_parquet(
        output_dir / "drugbank_drug_interactions.parquet"
    )
    assert drugs.columns == [
        "drugbank_id",
        "all_drugbank_ids",
        "name",
        "drug_type",
        "groups",
    ]
    assert partners["known_action"].to_list() == ["no"]
    assert partners["uniprot_ids"].to_list() == [["P12345"]]
    assert interactions.select(
        "source_drugbank_id", "target_drugbank_id"
    ).row(0) == ("DB00001", "DB00002")

def test_parser_limit_is_explicit_and_deterministic(tmp_path: Path) -> None:
    xml_path = _write_fixture(tmp_path / "drugbank.xml")

    records = list(iter_drugbank_records(xml_path, limit=1))

    assert len(records) == 1
    assert records[0].primary_drugbank_id == "DB00001"

from pathlib import Path

import numpy as np

import polars as pl

from evi_hgt.preprocessing.raw_source.canonical_facts import (
    EdgeComponentSpec,
    build_node_index,
    canonicalize_edge_partition,
)

from evi_hgt.preprocessing.raw_source.drugbank_merge import (
    build_conservative_drug_map,
    classify_drugbank_pair_description_match,
)

from evi_hgt.preprocessing.raw_source.graph_builder import build_csr_index

from evi_hgt.preprocessing.raw_source.phase3_schema import OverlapCategory

def _graph_properties(direct: list[str], rank: int = 1) -> dict[str, object]:
    return {
        "sources": {"direct": direct, "indirect": []},
        "expression_rank": rank,
        "call_quality": "gold",
    }

def _node_index() -> pl.DataFrame:
    nodes = {
        "nodes/anatomy.parquet": pl.DataFrame(
            {"id": ["ANA:1"], "label": ["ANA"], "properties": [{}]}
        ),
        "nodes/gene.parquet": pl.DataFrame(
            {
                "id": ["GEN:1", "GEN:2"],
                "label": ["GEN", "GEN"],
                "properties": [{}, {}],
            }
        ),
        "nodes/drug.parquet": pl.DataFrame(
            {
                "id": ["CHEMBL1", "CHEMBL2"],
                "label": ["DRG", "DRG"],
                "properties": [{}, {}],
            }
        ),
    }
    return build_node_index(nodes)

def test_fact_identity_is_separate_from_evidence_identity() -> None:
    frame = pl.DataFrame(
        [
            {
                "from": "ANA:1",
                "to": "GEN:1",
                "label": "ANA-GEN",
                "relation": "EXPRESSION_PRESENT",
                "undirected": True,
                "properties": _graph_properties(["BGEE"], rank=1),
            },
            {
                "from": "ANA:1",
                "to": "GEN:1",
                "label": "ANA-GEN",
                "relation": "EXPRESSION_PRESENT",
                "undirected": True,
                "properties": _graph_properties(["BGEE"], rank=2),
            },
        ]
    )
    spec = EdgeComponentSpec(
        relative_path="edges/anatomy_gene.parquet",
        source_type="ANA",
        target_type="GEN",
    )

    facts, evidence = canonicalize_edge_partition(frame, _node_index(), spec)

    assert facts.height == 1
    assert evidence.height == 2
    assert evidence["canonical_fact_id"].n_unique() == 1
    assert facts["aggregated_observation_count"].to_list() == [2]

def test_edge_counter_fields_are_relation_specific_not_dda_unsuitability() -> None:
    frame = pl.DataFrame(
        [
            {
                "from": "ANA:1",
                "to": "GEN:1",
                "label": "ANA-GEN",
                "relation": "EXPRESSION_ABSENT",
                "undirected": True,
                "properties": _graph_properties(["BGEE"]),
            }
        ]
    )
    spec = EdgeComponentSpec(
        relative_path="edges/anatomy_gene.parquet",
        source_type="ANA",
        target_type="GEN",
    )

    _, evidence = canonicalize_edge_partition(frame, _node_index(), spec)

    assert evidence["edge_counter_evidence_observed"].to_list() == [True]
    assert evidence["edge_support_observed"].to_list() == [False]
    assert not any("unsuit" in column for column in evidence.columns)

def test_synergistic_interaction_is_context_not_observed_polarity() -> None:
    frame = pl.DataFrame(
        [
            {
                "from": "CHEMBL1",
                "to": "CHEMBL2",
                "label": "DRG-DRG",
                "relation": "SYNERGISTIC_INTERACTION",
                "undirected": True,
                "properties": {"sources": {"direct": ["PIPELINE"], "indirect": []}},
            }
        ]
    )
    spec = EdgeComponentSpec(
        relative_path="edges/drug_drug.parquet",
        source_type="DRG",
        target_type="DRG",
    )

    _, evidence = canonicalize_edge_partition(frame, _node_index(), spec)

    assert evidence["edge_support_observed"].to_list() == [False]
    assert evidence["edge_counter_evidence_observed"].to_list() == [False]
    assert evidence["edge_evidence_proposition"].to_list() == [None]
    assert evidence["properties"].to_list()[0]["sources"]["direct"] == ["PIPELINE"]

def test_same_type_undirected_fact_has_one_canonical_orientation() -> None:
    frame = pl.DataFrame(
        [
            {
                "from": "CHEMBL2",
                "to": "CHEMBL1",
                "label": "DRG-DRG",
                "relation": "PARENT",
                "undirected": True,
                "properties": {"sources": {"direct": [], "indirect": []}},
            }
        ]
    )
    spec = EdgeComponentSpec(
        relative_path="edges/drug_drug.parquet",
        source_type="DRG",
        target_type="DRG",
    )

    facts, _ = canonicalize_edge_partition(frame, _node_index(), spec)

    assert facts["source_node_id"].to_list() == ["CHEMBL1"]
    assert facts["target_node_id"].to_list() == ["CHEMBL2"]

def test_conservative_drug_map_does_not_choose_ambiguous_chembl() -> None:
    drugs = pl.DataFrame({"drugbank_id": ["DB1", "DB2", "DB3"]})
    external = pl.DataFrame(
        {
            "drugbank_id": ["DB1", "DB2", "DB2"],
            "resource": ["ChEMBL", "ChEMBL", "ChEMBL"],
            "identifier": ["CHEMBL1", "CHEMBL2", "CHEMBL22"],
        }
    )

    mapping = build_conservative_drug_map(drugs.lazy(), external.lazy()).sort(
        "drugbank_id"
    )

    assert mapping["canonical_drug_id"].to_list() == [
        "CHEMBL1",
        "DRUGBANK:DB2",
        "DRUGBANK:DB3",
    ]
    assert mapping["mapping_status"].to_list() == [
        "UNIQUE_CHEMBL",
        "AMBIGUOUS_CHEMBL",
        "DRUGBANK_ONLY",
    ]

def test_drugbank_match_is_probable_without_shared_version_alignment() -> None:
    decision = classify_drugbank_pair_description_match(shared_version=None)

    assert decision.category is OverlapCategory.PROBABLE_DUPLICATE
    assert decision.independent_vote_eligible is False

def test_csr_expands_undirected_traversal_without_duplicating_fact(
    tmp_path: Path,
) -> None:
    facts = pl.DataFrame(
        {
            "canonical_fact_id": [5, 6],
            "source_global_id": [0, 1],
            "target_global_id": [1, 2],
            "source_node_type": ["DRG", "GEN"],
            "target_node_type": ["GEN", "GEN"],
            "relation": ["TARGET", "INTERACTS_WITH"],
            "undirected": [True, False],
        }
    )
    metadata = build_csr_index([facts], node_count=3, output_dir=tmp_path)
    indptr = np.load(tmp_path / "indptr.npy", mmap_mode="r")
    indices = np.load(tmp_path / "indices.npy", mmap_mode="r")
    fact_ids = np.load(tmp_path / "fact_ids.npy", mmap_mode="r")

    assert metadata["traversal_edge_count"] == 3
    assert indices[indptr[0] : indptr[1]].tolist() == [1]
    assert set(indices[indptr[1] : indptr[2]].tolist()) == {0, 2}
    assert fact_ids.tolist().count(5) == 2
    assert fact_ids.tolist().count(6) == 1

import pytest

from evi_hgt.preprocessing.raw_source.evidence_overlap import (
    EvidenceIdentity,
    classify_overlap,
    independent_vote_eligible,
)

from evi_hgt.preprocessing.raw_source.phase3_schema import OverlapCategory

from evi_hgt.preprocessing.raw_source.provenance import ProvenanceKind, classify_provenance_token

@pytest.mark.parametrize(
    ("category", "eligible"),
    [
        (OverlapCategory.EXACT_DUPLICATE, False),
        (OverlapCategory.PROBABLE_DUPLICATE, False),
        (OverlapCategory.SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE, True),
        (OverlapCategory.UNRESOLVED_OVERLAP, False),
    ],
)
def test_only_verified_independent_evidence_votes(
    category: OverlapCategory, eligible: bool
) -> None:
    assert independent_vote_eligible(category) is eligible

@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("DRUG_BANK", ProvenanceKind.DATABASE),
        ("PMID:12345678", ProvenanceKind.PMID),
        ("10.1038/s41586-024-00000-0", ProvenanceKind.DOI),
        ("GO_REF:0000002", ProvenanceKind.GO_REF),
        ("NCT01234567", ProvenanceKind.RECORD_IDENTIFIER),
        ("odd:unverified", ProvenanceKind.UNKNOWN_RAW),
    ],
)
def test_provenance_classification_is_conservative_and_raw_preserving(
    raw: str, kind: ProvenanceKind
) -> None:
    token = classify_provenance_token(raw)

    assert token.kind is kind
    assert token.raw == raw

def test_shared_database_name_does_not_prove_independence() -> None:
    left = EvidenceIdentity(
        source_release=None,
        source_record_id=None,
        reference_ids=(),
        source_membership=("DRUG_BANK",),
        semantic_fingerprint="same",
    )
    right = EvidenceIdentity(
        source_release=None,
        source_record_id=None,
        reference_ids=(),
        source_membership=("DRUG_BANK",),
        semantic_fingerprint="same",
    )

    decision = classify_overlap(left, right)

    assert decision.category is OverlapCategory.PROBABLE_DUPLICATE
    assert decision.independent_vote_eligible is False

def test_repeated_pmid_is_not_an_independent_vote() -> None:
    left = EvidenceIdentity(
        source_release="release-a",
        source_record_id="record-1",
        reference_ids=("PMID:1",),
        source_membership=("SOURCE_A",),
        semantic_fingerprint="same",
    )
    right = EvidenceIdentity(
        source_release="release-b",
        source_record_id="record-2",
        reference_ids=("PMID:1",),
        source_membership=("SOURCE_B",),
        semantic_fingerprint="same",
    )

    decision = classify_overlap(left, right)

    assert decision.category is OverlapCategory.UNRESOLVED_OVERLAP
    assert decision.independent_vote_eligible is False

@pytest.mark.parametrize(
    ("left_references", "right_references", "left_sources", "right_sources"),
    [
        (("10.1000/shared",), ("10.1000/shared",), ("AGGREGATOR_A",), ("AGGREGATOR_B",)),
        ((), (), ("DATABASE_A",), ("DATABASE_B",)),
    ],
)
def test_doi_or_database_name_multiplicity_is_not_independent(
    left_references: tuple[str, ...],
    right_references: tuple[str, ...],
    left_sources: tuple[str, ...],
    right_sources: tuple[str, ...],
) -> None:
    left = EvidenceIdentity(
        source_release=None,
        source_record_id="record-1",
        reference_ids=left_references,
        source_membership=left_sources,
        semantic_fingerprint="same",
    )
    right = EvidenceIdentity(
        source_release=None,
        source_record_id="record-2",
        reference_ids=right_references,
        source_membership=right_sources,
        semantic_fingerprint="same",
    )

    decision = classify_overlap(left, right)

    assert decision.category is OverlapCategory.UNRESOLVED_OVERLAP
    assert decision.independent_vote_eligible is False

def test_only_verified_distinct_records_with_disjoint_chains_are_independent() -> None:
    left = EvidenceIdentity(
        source_release="release-a",
        source_record_id="record-1",
        reference_ids=("PMID:1",),
        source_membership=("SOURCE_A",),
        semantic_fingerprint="same",
    )
    right = EvidenceIdentity(
        source_release="release-b",
        source_record_id="record-2",
        reference_ids=("PMID:2",),
        source_membership=("SOURCE_B",),
        semantic_fingerprint="same",
    )

    unresolved = classify_overlap(left, right)
    independent = classify_overlap(
        left,
        right,
        verified_distinct_records=True,
        verified_disjoint_provenance_chains=True,
    )

    assert unresolved.category is OverlapCategory.UNRESOLVED_OVERLAP
    assert independent.category is (
        OverlapCategory.SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE
    )
    assert independent.independent_vote_eligible is True

from pathlib import Path

import polars as pl

import pytest

from evi_hgt.preprocessing.raw_source.dda_supervision import (
    DDASupervisionError,
    build_dda_supervision,
    supervision_statistics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DDA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "optimuskg"
    / "doi_10_7910_DVN_IYNGEV"
    / "2.0"
    / "edges"
    / "drug_disease.parquet"
)

def _properties(
    *,
    direct: list[str],
    phase: float | None = None,
    references: list[str] | None = None,
) -> dict[str, object]:
    return {
        "sources": {"direct": direct, "indirect": []},
        "structure_id": None,
        "drug_disease_id": None,
        "reference_ids": references or [],
        "highest_clinical_trial_phase": phase,
    }

@pytest.fixture
def toy_dda() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "from": "DRG:DC",
                "to": "DIS:1",
                "relation": "INDICATION",
                "undirected": True,
                "properties": _properties(direct=["DRUG_CENTRAL"]),
            },
            {
                "from": "DRG:PHASE",
                "to": "DIS:2",
                "relation": "INDICATION",
                "undirected": True,
                "properties": _properties(direct=["OPEN_TARGETS"], phase=2.0),
            },
            {
                "from": "DRG:EARLY",
                "to": "DIS:3",
                "relation": "INDICATION",
                "undirected": True,
                "properties": _properties(direct=["OPEN_TARGETS"], phase=1.0),
            },
            {
                "from": "DRG:CONTRA",
                "to": "DIS:4",
                "relation": "CONTRAINDICATION",
                "undirected": True,
                "properties": _properties(direct=["DRUG_CENTRAL"]),
            },
            {
                "from": "DRG:OFF",
                "to": "DIS:5",
                "relation": "OFF_LABEL_USE",
                "undirected": True,
                "properties": _properties(direct=["DRUG_CENTRAL"]),
            },
        ]
    )

def test_policy_b_uses_nullable_independent_channels(toy_dda: pl.DataFrame) -> None:
    artifact = build_dda_supervision(toy_dda, policy="B")
    rows = {row["drug_id"]: row for row in artifact.to_dicts()}

    assert rows["DRG:DC"]["support_target"] == 1.0
    assert rows["DRG:PHASE"]["support_target"] == 1.0
    assert rows["DRG:EARLY"]["support_target"] is None
    assert rows["DRG:EARLY"]["support_mask"] is False
    assert rows["DRG:CONTRA"]["unsuitability_target"] == 1.0
    assert rows["DRG:CONTRA"]["unsuitability_mask"] is True
    assert rows["DRG:CONTRA"]["support_target"] is None
    assert rows["DRG:CONTRA"]["unsuitability_observation_incomplete"] is True
    assert "four_value_label" not in artifact.columns

def test_off_label_is_context_only_and_never_supervised(
    toy_dda: pl.DataFrame,
) -> None:
    row = (
        build_dda_supervision(toy_dda, policy="B")
        .filter(pl.col("raw_relation") == "OFF_LABEL_USE")
        .row(0, named=True)
    )

    assert row["context_only"] is True
    assert row["support_target"] is None and row["support_mask"] is False
    assert row["unsuitability_target"] is None
    assert row["unsuitability_mask"] is False

def test_absent_contraindication_never_becomes_suitability_zero(
    toy_dda: pl.DataFrame,
) -> None:
    indication_rows = build_dda_supervision(toy_dda).filter(
        pl.col("raw_relation") == "INDICATION"
    )

    assert indication_rows["unsuitability_target"].null_count() == (
        indication_rows.height
    )
    assert indication_rows["unsuitability_mask"].sum() == 0

def test_policy_membership_is_separate_from_active_policy_mask(
    toy_dda: pl.DataFrame,
) -> None:
    early_b = build_dda_supervision(toy_dda, policy="B").filter(
        pl.col("drug_id") == "DRG:EARLY"
    ).row(0, named=True)
    early_c = build_dda_supervision(toy_dda, policy="C").filter(
        pl.col("drug_id") == "DRG:EARLY"
    ).row(0, named=True)

    assert early_b["policy_c_member"] is True
    assert early_b["support_mask"] is False
    assert early_c["support_mask"] is True
    assert early_c["support_target"] == 1.0

def test_duplicate_canonical_pairs_fail_closed(toy_dda: pl.DataFrame) -> None:
    duplicate = pl.concat([toy_dda, toy_dda.head(1)])

    with pytest.raises(DDASupervisionError, match="canonical pair"):
        build_dda_supervision(duplicate)

from evi_hgt.preprocessing.raw_source.drugbank_merge import build_drugbank_partner_facts, audit_drugbank_interaction_overlap


def test_drugbank_partner_merge_reuses_fact_and_adds_only_local_supplement() -> None:
    drug_map = pl.DataFrame(
        {
            "drugbank_id": ["DB1", "DB2", "DB3"],
            "canonical_drug_id": ["CHEMBL1", "DRUGBANK:DB2", "DRUGBANK:DB3"],
            "mapping_status": ["UNIQUE_CHEMBL", "DRUGBANK_ONLY", "DRUGBANK_ONLY"],
            "graph_membership": [
                "OPTIMUSKG_MERGED",
                "DRUGBANK_SUPPLEMENT",
                "DRUGBANK_SUPPLEMENT",
            ],
        }
    )
    gene_nodes = pl.DataFrame(
        {
            "id": ["ENSG1", "ENSG2"],
            "properties": [
                '{"associated_proteins":[{"id":"P1","source":"uniprot_swissprot"}]}',
                '{"associated_proteins":[{"id":"P2","source":"uniprot_swissprot"}]}',
            ],
        }
    )
    node_index = pl.DataFrame(
        {
            "global_node_id": [0, 1, 2, 3, 4],
            "type_local_id": [0, 1, 2, 0, 1],
            "node_type": ["DRG", "DRG", "DRG", "GEN", "GEN"],
            "node_id": [
                "CHEMBL1",
                "DRUGBANK:DB2",
                "DRUGBANK:DB3",
                "ENSG1",
                "ENSG2",
            ],
            "source_component_path": ["test"] * 5,
        }
    ).with_columns(
        pl.col("global_node_id").cast(pl.UInt32),
        pl.col("type_local_id").cast(pl.UInt32),
    )
    existing_facts = pl.DataFrame(
        {
            "canonical_fact_id": [10],
            "canonical_fact_key": ["DRG\u001fCHEMBL1\u001fTARGET\u001fGEN\u001fENSG1\u001fDIRECTED"],
            "source_node_type": ["DRG"],
            "source_node_id": ["CHEMBL1"],
            "source_global_id": [0],
            "target_node_type": ["GEN"],
            "target_node_id": ["ENSG1"],
            "target_global_id": [3],
            "relation": ["TARGET"],
            "undirected": [False],
            "aggregated_observation_count": [1],
            "source_component_path": ["edges/drug_gene.parquet"],
        }
    ).with_columns(pl.col("canonical_fact_id").cast(pl.UInt64))
    partners = pl.DataFrame(
        {
            "drugbank_id": ["DB1", "DB2", "DB3"],
            "category": ["target", "enzyme", "target"],
            "partner_id": ["1", "2", "3"],
            "name": ["one", "two", "three"],
            "organism": ["Humans", "Humans", "Humans"],
            "actions": [["inhibitor"], ["substrate"], []],
            "known_action": ["yes", "no", "unknown"],
            "reference_ids": [["PMID:1"], ["PMID:2"], []],
            "polypeptide_id": ["P1", "P2", None],
            "polypeptide_source": ["Swiss-Prot", "Swiss-Prot", None],
            "polypeptide_name": ["one", "two", None],
            "gene_name": ["G1", "G2", None],
            "polypeptide_organism": ["Humans", "Humans", None],
            "uniprot_ids": [["P1"], ["P2"], []],
        }
    )

    merge = build_drugbank_partner_facts(
        partners,
        drug_map,
        gene_nodes,
        node_index,
        existing_facts,
        fact_id_offset=20,
        evidence_id_offset=30,
    )

    assert merge.local_only_facts.height == 1
    assert merge.local_only_facts["canonical_fact_id"].to_list() == [20]
    assert merge.local_only_facts["relation"].to_list() == ["ENZYME"]
    assert merge.evidence_records.height == 2
    assert set(merge.evidence_records["canonical_fact_id"]) == {10, 20}
    overlap = merge.membership.filter(pl.col("optimuskg_membership"))
    assert overlap.height == 1
    assert overlap["overlap_category"].to_list() == ["UNRESOLVED_OVERLAP"]
    assert overlap["independent_vote_eligible"].to_list() == [False]
    assert merge.summary["mapped_partner_observation_count"] == 2
    assert merge.summary["unmapped_partner_observation_count"] == 1
    assert merge.summary["existing_canonical_fact_overlap_count"] == 1
    assert merge.summary["local_only_canonical_fact_count"] == 1

def test_drugbank_interaction_overlap_is_probable_without_release_alignment(
    tmp_path: Path,
) -> None:
    interactions = tmp_path / "interactions.parquet"
    drug_drug = tmp_path / "drug_drug.parquet"
    pl.DataFrame(
        {
            "source_drugbank_id": ["DB2", "DB1", "DB1"],
            "target_drugbank_id": ["DB1", "DB2", "DB3"],
            "description": ["same text", "same   text", "different"],
        }
    ).write_parquet(interactions)
    pl.DataFrame(
        {
            "from": ["CHEMBL1", "CHEMBL1"],
            "to": ["CHEMBL2", "DRUGBANK:DB3"],
            "relation": ["SYNERGISTIC_INTERACTION", "SYNERGISTIC_INTERACTION"],
            "properties": [
                '{"sources":{"direct":["DRUG_BANK"],"indirect":[]},"interaction_description":"same text"}',
                '{"sources":{"direct":["DRUG_BANK"],"indirect":[]},"interaction_description":"other"}',
            ],
        }
    ).write_parquet(drug_drug)
    mapping = pl.DataFrame(
        {
            "drugbank_id": ["DB1", "DB2", "DB3"],
            "canonical_drug_id": ["CHEMBL1", "CHEMBL2", "DRUGBANK:DB3"],
        }
    )

    audit = audit_drugbank_interaction_overlap(
        interactions,
        mapping,
        drug_drug,
        shared_release_alignment=None,
    )

    assert audit["local_unique_pair_count"] == 2
    assert audit["optimuskg_drugbank_pair_count"] == 2
    assert audit["pair_overlap_count"] == 2
    assert audit["PROBABLE_DUPLICATE"] == 1
    assert audit["UNRESOLVED_OVERLAP"] == 1
    assert audit["EXACT_DUPLICATE"] == 0
    assert audit["SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE"] == 0
    assert audit["independent_vote_eligible"] == 0
