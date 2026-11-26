"""Conservative DrugBank mappings and overlap decisions for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from evi_hgt.preprocessing.raw_source.evidence_overlap import OverlapDecision
from evi_hgt.preprocessing.raw_source.phase3_schema import OverlapCategory


def build_conservative_drug_map(
    drugs: pl.LazyFrame, external_identifiers: pl.LazyFrame
) -> pl.DataFrame:
    """Use a ChEMBL ID only when the local DrugBank crosswalk is unique."""

    chembl = (
        external_identifiers.filter(pl.col("resource") == "ChEMBL")
        .group_by("drugbank_id")
        .agg(pl.col("identifier").drop_nulls().unique().sort().alias("chembl_ids"))
    )
    return (
        drugs.select("drugbank_id")
        .unique()
        .join(chembl, on="drugbank_id", how="left")
        .with_columns(
            pl.col("chembl_ids").list.len().fill_null(0).alias("chembl_id_count")
        )
        .with_columns(
            pl.when(pl.col("chembl_id_count") == 1)
            .then(pl.col("chembl_ids").list.first())
            .otherwise(pl.lit("DRUGBANK:") + pl.col("drugbank_id"))
            .alias("canonical_drug_id"),
            pl.when(pl.col("chembl_id_count") == 1)
            .then(pl.lit("UNIQUE_CHEMBL"))
            .when(pl.col("chembl_id_count") > 1)
            .then(pl.lit("AMBIGUOUS_CHEMBL"))
            .otherwise(pl.lit("DRUGBANK_ONLY"))
            .alias("mapping_status"),
        )
        .select(
            "drugbank_id",
            "canonical_drug_id",
            "chembl_id_count",
            "mapping_status",
        )
        .collect(engine="streaming")
    )


def classify_drugbank_pair_description_match(
    *, shared_version: str | None
) -> OverlapDecision:
    """Classify a pair-plus-description match without overstating identity."""

    if shared_version:
        return OverlapDecision(
            category=OverlapCategory.EXACT_DUPLICATE,
            independent_vote_eligible=False,
            decision_basis=(
                "same canonical pair, normalized description, and pinned DrugBank "
                f"source version {shared_version}"
            ),
        )
    return OverlapDecision(
        category=OverlapCategory.PROBABLE_DUPLICATE,
        independent_vote_eligible=False,
        decision_basis=(
            "same canonical pair and normalized interaction description without "
            "verified DrugBank release alignment"
        ),
        unresolved_reason="shared source-version fingerprint unavailable",
    )


def build_formal_drugbank_node_map(
    drugs: pl.LazyFrame,
    external_identifiers: pl.LazyFrame,
    *,
    existing_drug_ids: set[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create conservative graph identities and explicit supplement nodes.

    An ambiguous ChEMBL crosswalk never chooses a convenient identifier.  It
    remains a DrugBank-namespaced node until a scientifically reviewed
    crosswalk resolves the ambiguity.
    """

    mapping = build_conservative_drug_map(drugs, external_identifiers).with_columns(
        pl.when(pl.col("canonical_drug_id").is_in(sorted(existing_drug_ids)))
        .then(pl.lit("OPTIMUSKG_MERGED"))
        .otherwise(pl.lit("DRUGBANK_SUPPLEMENT"))
        .alias("graph_membership")
    )
    supplements = (
        mapping.filter(pl.col("graph_membership") == "DRUGBANK_SUPPLEMENT")
        .select(
            pl.col("canonical_drug_id").alias("node_id"),
            pl.lit("DRG").alias("node_type"),
            "drugbank_id",
            "mapping_status",
            "chembl_id_count",
            pl.lit("drugbank/local_drug_nodes").alias("source_component_path"),
        )
        .unique(subset=["node_id"], keep="first")
        .sort("node_id")
    )
    return mapping.sort("drugbank_id"), supplements


@dataclass(frozen=True, slots=True)
class DrugBankPartnerMerge:
    """Formal conservative DrugBank Drug-Gene merge outputs."""

    local_only_facts: pl.DataFrame
    evidence_records: pl.DataFrame
    membership: pl.DataFrame
    unmapped_partners: pl.DataFrame
    summary: dict[str, int | str | bool]


def _normalized_description(expression: pl.Expr) -> pl.Expr:
    return expression.str.replace_all(r"\s+", " ").str.strip_chars()


def audit_drugbank_interaction_overlap(
    interactions_path: str | Path,
    drug_map: pl.DataFrame,
    optimuskg_drug_drug_path: str | Path,
    *,
    shared_release_alignment: str | None,
) -> dict[str, int | str | bool | None]:
    """Count conservative DrugBank/OptimusKG Drug-Drug overlaps.

    Pair-plus-normalized-description identity is only probable without a shared
    DrugBank release fingerprint.  Pair-only matches remain unresolved.
    """

    from pathlib import Path

    required_map = {"drugbank_id", "canonical_drug_id"}
    if missing := required_map - set(drug_map.columns):
        raise ValueError(f"Drug map lacks fields: {sorted(missing)}")
    mapping = drug_map.select("drugbank_id", "canonical_drug_id").lazy()
    low_map = mapping.select(
        pl.col("drugbank_id").alias("_drugbank_low"),
        pl.col("canonical_drug_id").alias("from"),
    )
    high_map = mapping.select(
        pl.col("drugbank_id").alias("_drugbank_high"),
        pl.col("canonical_drug_id").alias("to"),
    )
    local = (
        pl.scan_parquet(Path(interactions_path))
        .with_columns(
            pl.min_horizontal(
                "source_drugbank_id", "target_drugbank_id"
            ).alias("_drugbank_low"),
            pl.max_horizontal(
                "source_drugbank_id", "target_drugbank_id"
            ).alias("_drugbank_high"),
        )
        .join(low_map, on="_drugbank_low", how="left")
        .join(high_map, on="_drugbank_high", how="left")
        .filter(pl.col("from").is_not_null() & pl.col("to").is_not_null())
        .with_columns(
            _normalized_description(pl.col("description")).alias(
                "normalized_description"
            )
        )
        .select("from", "to", "normalized_description")
        .unique()
        .collect(engine="streaming")
    )
    kg_scan = pl.scan_parquet(Path(optimuskg_drug_drug_path))
    properties_dtype = kg_scan.collect_schema()["properties"]
    if properties_dtype == pl.String:
        direct = (
            pl.col("properties")
            .str.json_path_match("$.sources.direct")
            .str.json_decode(pl.List(pl.String))
        )
        description = pl.col("properties").str.json_path_match(
            "$.interaction_description"
        )
    elif isinstance(properties_dtype, pl.Struct):
        direct = (
            pl.col("properties")
            .struct.field("sources")
            .struct.field("direct")
        )
        description = pl.col("properties").struct.field(
            "interaction_description"
        )
    else:
        raise ValueError("Unsupported Drug-Drug properties representation")
    kg = (
        kg_scan.with_columns(
            direct.alias("_direct_sources"),
            description.alias("_description"),
        )
        .filter(
            (pl.col("relation") == "SYNERGISTIC_INTERACTION")
            & pl.col("_direct_sources").list.contains("DRUG_BANK")
        )
        .with_columns(
            _normalized_description(pl.col("_description")).alias(
                "normalized_description"
            )
        )
        .select("from", "to", "normalized_description")
        .unique()
        .collect(engine="streaming")
    )
    local_pairs = local.select("from", "to").unique()
    kg_pairs = kg.select("from", "to").unique()
    pair_overlap = local_pairs.join(
        kg_pairs, on=["from", "to"], how="inner"
    ).height
    fingerprint_overlap = local.join(
        kg,
        on=["from", "to", "normalized_description"],
        how="inner",
    ).height
    exact = fingerprint_overlap if shared_release_alignment else 0
    probable = 0 if shared_release_alignment else fingerprint_overlap
    unresolved = pair_overlap - fingerprint_overlap
    if unresolved < 0:
        raise ValueError("DrugBank fingerprint overlap exceeds pair overlap")
    return {
        "policy": "PAIR_PLUS_NORMALIZED_DESCRIPTION_FAIL_CLOSED",
        "shared_release_alignment": shared_release_alignment,
        "local_unique_pair_count": local_pairs.height,
        "optimuskg_drugbank_pair_count": kg_pairs.height,
        "pair_overlap_count": pair_overlap,
        "normalized_description_fingerprint_overlap_count": fingerprint_overlap,
        "EXACT_DUPLICATE": exact,
        "PROBABLE_DUPLICATE": probable,
        "SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE": 0,
        "UNRESOLVED_OVERLAP": unresolved,
        "independent_vote_eligible": 0,
        "semantic_degraded_synergy_context_only": True,
    }


def _unique_uniprot_gene_map(gene_nodes: pl.DataFrame) -> pl.DataFrame:
    required = {"id", "properties"}
    if missing := required - set(gene_nodes.columns):
        raise ValueError(f"Gene nodes lack required fields: {sorted(missing)}")
    dtype = gene_nodes.schema["properties"]
    proteins_dtype = pl.List(
        pl.Struct({"id": pl.String, "source": pl.String})
    )
    if dtype == pl.String:
        proteins = (
            pl.col("properties")
            .str.json_path_match("$.associated_proteins")
            .str.json_decode(proteins_dtype)
        )
    elif isinstance(dtype, pl.Struct) and "associated_proteins" in {
        field.name for field in dtype.fields
    }:
        proteins = pl.col("properties").struct.field("associated_proteins")
    else:
        raise ValueError(
            "Gene properties cannot provide verified associated_proteins"
        )
    exploded = (
        gene_nodes.select(
            pl.col("id").alias("canonical_gene_id"),
            proteins.alias("associated_proteins"),
        )
        .explode("associated_proteins", empty_as_null=True)
        .unnest("associated_proteins")
        .filter(
            pl.col("id").is_not_null()
            & pl.col("source").str.to_lowercase().str.starts_with("uniprot")
        )
        .select(
            pl.col("id").alias("uniprot_id"),
            "canonical_gene_id",
        )
        .unique()
    )
    return (
        exploded.group_by("uniprot_id")
        .agg(
            pl.col("canonical_gene_id").unique().sort().alias("gene_ids")
        )
        .with_columns(
            pl.col("gene_ids").list.len().alias("gene_id_count")
        )
        .filter(pl.col("gene_id_count") == 1)
        .select(
            "uniprot_id",
            pl.col("gene_ids").list.first().alias("canonical_gene_id"),
        )
    )


def _partner_property_columns() -> list[str]:
    return [
        "drugbank_id",
        "category",
        "partner_id",
        "name",
        "organism",
        "actions",
        "known_action",
        "reference_ids",
        "polypeptide_id",
        "polypeptide_source",
        "polypeptide_name",
        "gene_name",
        "polypeptide_organism",
        "uniprot_ids",
    ]


def build_drugbank_partner_facts(
    partners: pl.DataFrame,
    drug_map: pl.DataFrame,
    gene_nodes: pl.DataFrame,
    node_index: pl.DataFrame,
    existing_facts: pl.DataFrame,
    *,
    fact_id_offset: int,
    evidence_id_offset: int,
) -> DrugBankPartnerMerge:
    """Merge only uniquely mapped structured DrugBank Drug-Gene records.

    Same canonical OptimusKG facts are reused.  Because the shared DrugBank
    release/record crosswalk is unavailable, overlap is fail-closed as
    ``UNRESOLVED_OVERLAP`` and cannot add an independent vote.
    """

    required_partner = set(_partner_property_columns())
    if missing := required_partner - set(partners.columns):
        raise ValueError(f"DrugBank partners lack fields: {sorted(missing)}")
    if {"drugbank_id", "canonical_drug_id"} - set(drug_map.columns):
        raise ValueError("DrugBank node map is incomplete")
    unique_genes = _unique_uniprot_gene_map(gene_nodes)
    indexed = partners.with_row_index("_drugbank_source_row")
    resolved_genes = (
        indexed.select("_drugbank_source_row", "uniprot_ids")
        .explode("uniprot_ids", empty_as_null=True)
        .rename({"uniprot_ids": "uniprot_id"})
        .join(unique_genes, on="uniprot_id", how="left")
        .group_by("_drugbank_source_row")
        .agg(
            pl.col("canonical_gene_id")
            .drop_nulls()
            .unique()
            .sort()
            .alias("resolved_gene_ids")
        )
        .with_columns(
            pl.col("resolved_gene_ids").list.len().alias("resolved_gene_count")
        )
    )
    work = (
        indexed.join(
            drug_map.select("drugbank_id", "canonical_drug_id"),
            on="drugbank_id",
            how="left",
            validate="m:1",
        )
        .join(resolved_genes, on="_drugbank_source_row", how="left", validate="1:1")
        .with_columns(
            pl.col("resolved_gene_count").fill_null(0),
            pl.when(pl.col("resolved_gene_count") == 1)
            .then(pl.col("resolved_gene_ids").list.first())
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("canonical_gene_id"),
            pl.col("category").str.to_uppercase().alias("relation"),
        )
    )
    allowed_relations = {"TARGET", "ENZYME", "CARRIER", "TRANSPORTER"}
    unexpected = set(work["relation"].drop_nulls().unique()) - allowed_relations
    if unexpected:
        raise ValueError(f"Unexpected DrugBank partner categories: {sorted(unexpected)}")
    mapped = work.filter(
        pl.col("canonical_drug_id").is_not_null()
        & pl.col("canonical_gene_id").is_not_null()
        & (pl.col("resolved_gene_count") == 1)
    ).with_columns(
        pl.concat_str(
            pl.lit("DRG"),
            "canonical_drug_id",
            "relation",
            pl.lit("GEN"),
            "canonical_gene_id",
            pl.lit("DIRECTED"),
            separator="\u001f",
        ).alias("canonical_fact_key")
    )
    unmapped = work.filter(
        pl.col("canonical_drug_id").is_null()
        | (pl.col("resolved_gene_count") != 1)
    ).select(
        "_drugbank_source_row",
        "drugbank_id",
        "category",
        "partner_id",
        "uniprot_ids",
        "resolved_gene_count",
        pl.when(pl.col("canonical_drug_id").is_null())
        .then(pl.lit("UNMAPPED_DRUG"))
        .when(pl.col("resolved_gene_count") == 0)
        .then(pl.lit("UNMAPPED_GENE"))
        .otherwise(pl.lit("AMBIGUOUS_GENE"))
        .alias("unmapped_reason"),
    )

    existing = existing_facts.select(
        "canonical_fact_key", "canonical_fact_id"
    ).unique()
    mapped = mapped.join(
        existing.rename({"canonical_fact_id": "existing_canonical_fact_id"}),
        on="canonical_fact_key",
        how="left",
        validate="m:1",
    )
    local_keys = (
        mapped.filter(pl.col("existing_canonical_fact_id").is_null())
        .group_by("canonical_fact_key")
        .agg(
            pl.col("canonical_drug_id").first(),
            pl.col("canonical_gene_id").first(),
            pl.col("relation").first(),
            pl.len().cast(pl.UInt32).alias("aggregated_observation_count"),
        )
        .sort("canonical_fact_key")
        .with_columns(
            pl.Series(
                "canonical_fact_id",
                range(fact_id_offset, fact_id_offset + mapped.filter(pl.col("existing_canonical_fact_id").is_null())["canonical_fact_key"].n_unique()),
                dtype=pl.UInt64,
            )
        )
    )
    source_nodes = node_index.filter(pl.col("node_type") == "DRG").select(
        pl.col("node_id").alias("canonical_drug_id"),
        pl.col("global_node_id").alias("source_global_id"),
    )
    target_nodes = node_index.filter(pl.col("node_type") == "GEN").select(
        pl.col("node_id").alias("canonical_gene_id"),
        pl.col("global_node_id").alias("target_global_id"),
    )
    local_facts = (
        local_keys.join(source_nodes, on="canonical_drug_id", how="left", validate="m:1")
        .join(target_nodes, on="canonical_gene_id", how="left", validate="m:1")
        .with_columns(
            pl.lit("DRG").alias("source_node_type"),
            pl.lit("GEN").alias("target_node_type"),
            pl.lit(False).alias("undirected"),
            pl.lit("drugbank/local_partners").alias("source_component_path"),
        )
        .rename(
            {
                "canonical_drug_id": "source_node_id",
                "canonical_gene_id": "target_node_id",
            }
        )
        .select(
            "canonical_fact_id",
            "canonical_fact_key",
            "source_node_type",
            "source_node_id",
            "source_global_id",
            "target_node_type",
            "target_node_id",
            "target_global_id",
            "relation",
            "undirected",
            "aggregated_observation_count",
            "source_component_path",
        )
    )
    if local_facts.filter(
        pl.col("source_global_id").is_null() | pl.col("target_global_id").is_null()
    ).height:
        raise ValueError("Mapped DrugBank partner fact has an unmapped graph node")

    fact_lookup = pl.concat(
        [
            existing.rename({"canonical_fact_id": "resolved_canonical_fact_id"}),
            local_facts.select(
                "canonical_fact_key",
                pl.col("canonical_fact_id").alias("resolved_canonical_fact_id"),
            ),
        ]
    )
    mapped = mapped.join(
        fact_lookup,
        on="canonical_fact_key",
        how="left",
        validate="m:1",
    )
    property_columns = _partner_property_columns()
    evidence = (
        mapped.sort("_drugbank_source_row")
        .with_columns(
            pl.Series(
                "evidence_record_id",
                range(evidence_id_offset, evidence_id_offset + mapped.height),
                dtype=pl.UInt64,
            ),
            pl.struct(property_columns).struct.json_encode().alias("properties"),
            pl.lit(["DRUG_BANK_LOCAL"], dtype=pl.List(pl.String)).alias(
                "raw_direct_sources"
            ),
            pl.lit([], dtype=pl.List(pl.String)).alias("raw_indirect_sources"),
            pl.lit(False).alias("edge_support_observed"),
            pl.lit(False).alias("edge_counter_evidence_observed"),
            pl.lit(None, dtype=pl.String).alias("edge_evidence_proposition"),
            pl.when(pl.col("existing_canonical_fact_id").is_not_null())
            .then(pl.lit(OverlapCategory.UNRESOLVED_OVERLAP.value))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("overlap_category"),
            pl.lit(False).alias("independent_vote_eligible"),
            pl.lit("drugbank/local_partners").alias("source_component_path"),
        )
        .select(
            "evidence_record_id",
            pl.col("resolved_canonical_fact_id").cast(pl.UInt64).alias(
                "canonical_fact_id"
            ),
            "canonical_fact_key",
            "source_component_path",
            pl.col("_drugbank_source_row").alias("source_row_number"),
            "relation",
            "properties",
            "raw_direct_sources",
            "raw_indirect_sources",
            "edge_support_observed",
            "edge_counter_evidence_observed",
            "edge_evidence_proposition",
            "overlap_category",
            "independent_vote_eligible",
        )
    )
    membership = (
        mapped.group_by("canonical_fact_key")
        .agg(
            pl.col("resolved_canonical_fact_id").first().cast(pl.UInt64).alias(
                "canonical_fact_id"
            ),
            pl.len().alias("drugbank_observation_count"),
            pl.col("existing_canonical_fact_id").is_not_null().any().alias(
                "optimuskg_membership"
            ),
        )
        .with_columns(
            pl.lit(True).alias("drugbank_membership"),
            pl.when(pl.col("optimuskg_membership"))
            .then(pl.lit(OverlapCategory.UNRESOLVED_OVERLAP.value))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("overlap_category"),
            pl.lit(False).alias("independent_vote_eligible"),
            pl.when(pl.col("optimuskg_membership"))
            .then(
                pl.lit(
                    "same canonical fact; shared DrugBank release/record identity unavailable"
                )
            )
            .otherwise(pl.lit("local DrugBank-only supplement"))
            .alias("decision_basis"),
        )
        .sort("canonical_fact_key")
    )
    summary: dict[str, int | str | bool] = {
        "policy": "CONSERVATIVE_FACT_LEVEL_MERGE",
        "mapped_partner_observation_count": mapped.height,
        "unmapped_partner_observation_count": unmapped.height,
        "existing_canonical_fact_overlap_count": membership.filter(
            pl.col("optimuskg_membership")
        ).height,
        "local_only_canonical_fact_count": local_facts.height,
        "independent_vote_eligible_count": 0,
        "exact_original_drugbank_record_reconstruction_claimed": False,
    }
    return DrugBankPartnerMerge(
        local_only_facts=local_facts,
        evidence_records=evidence,
        membership=membership,
        unmapped_partners=unmapped,
        summary=summary,
    )
