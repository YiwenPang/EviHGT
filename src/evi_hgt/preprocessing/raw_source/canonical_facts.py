"""Canonical node/fact construction with separate aggregated evidence records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import polars as pl

from evi_hgt.preprocessing.raw_source.phase3_schema import OverlapCategory


NODE_TYPES = {
    "anatomy": "ANA",
    "biological_process": "BPO",
    "cellular_component": "CCO",
    "disease": "DIS",
    "drug": "DRG",
    "exposure": "EXP",
    "gene": "GEN",
    "molecular_function": "MFN",
    "pathway": "PWY",
    "phenotype": "PHE",
}


class CanonicalFactError(ValueError):
    """Raised when endpoint types or canonical identity cannot be verified."""


@dataclass(frozen=True, slots=True)
class EdgeComponentSpec:
    relative_path: str
    source_type: str
    target_type: str


def node_type_from_component(relative_path: str) -> str:
    path = Path(relative_path)
    if path.parent.as_posix() != "nodes" or path.suffix != ".parquet":
        raise CanonicalFactError(f"Not a node component: {relative_path}")
    try:
        return NODE_TYPES[path.stem]
    except KeyError as exc:
        raise CanonicalFactError(f"Unknown node component: {relative_path}") from exc


def edge_component_spec(relative_path: str) -> EdgeComponentSpec:
    path = Path(relative_path)
    if path.parent.as_posix() != "edges" or path.suffix != ".parquet":
        raise CanonicalFactError(f"Not an edge component: {relative_path}")
    matches = [
        (source, target)
        for source in NODE_TYPES
        for target in NODE_TYPES
        if path.stem == f"{source}_{target}"
    ]
    if len(matches) != 1:
        raise CanonicalFactError(f"Cannot resolve endpoint types: {relative_path}")
    source, target = matches[0]
    return EdgeComponentSpec(relative_path, NODE_TYPES[source], NODE_TYPES[target])


def _read_frame(value: pl.DataFrame | str | Path, columns: list[str]) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value.select([name for name in columns if name in value.columns])
    return pl.read_parquet(value, columns=columns)


def build_node_index(
    components: Mapping[str, pl.DataFrame | str | Path],
) -> pl.DataFrame:
    """Assign deterministic global and type-local integer node IDs."""

    partitions: list[pl.DataFrame] = []
    for relative_path in sorted(components):
        node_type = node_type_from_component(relative_path)
        frame = _read_frame(components[relative_path], ["id", "label"])
        if "id" not in frame.columns:
            raise CanonicalFactError(f"Node component lacks id: {relative_path}")
        if frame["id"].null_count():
            raise CanonicalFactError(f"Node component has null id: {relative_path}")
        if "label" in frame.columns:
            labels = set(frame["label"].drop_nulls().unique().to_list())
            if labels and labels != {node_type}:
                raise CanonicalFactError(
                    f"Node label mismatch for {relative_path}: {sorted(labels)}"
                )
        partitions.append(
            frame.select(pl.col("id").cast(pl.String).alias("node_id"))
            .with_columns(
                pl.lit(node_type).alias("node_type"),
                pl.lit(relative_path).alias("source_component_path"),
            )
        )
    if not partitions:
        raise CanonicalFactError("No node components supplied")
    result = pl.concat(partitions).sort(["node_type", "node_id"])
    if result.select("node_type", "node_id").is_duplicated().any():
        raise CanonicalFactError("Canonical typed node IDs must be unique")
    result = result.with_columns(
        (pl.col("node_id").cum_count().over("node_type") - 1)
        .cast(pl.UInt32)
        .alias("type_local_id")
    ).with_columns(
        pl.Series("global_node_id", range(result.height), dtype=pl.UInt32)
    )
    return result.select(
        "global_node_id",
        "type_local_id",
        "node_type",
        "node_id",
        "source_component_path",
    )


def augment_node_index(
    base_index: pl.DataFrame,
    supplemental_nodes: pl.DataFrame,
) -> pl.DataFrame:
    """Add verified local-only nodes and deterministically rebuild integer IDs.

    Existing OptimusKG membership has precedence when a supplement resolves to
    an already present typed identifier.
    """

    required_base = {"node_type", "node_id", "source_component_path"}
    required_supplement = {"node_type", "node_id", "source_component_path"}
    if missing := required_base - set(base_index.columns):
        raise CanonicalFactError(f"Base node index lacks {sorted(missing)}")
    if missing := required_supplement - set(supplemental_nodes.columns):
        raise CanonicalFactError(f"Supplement nodes lack {sorted(missing)}")
    base = base_index.select(
        "node_type", "node_id", "source_component_path"
    )
    if base.select("node_type", "node_id").is_duplicated().any():
        raise CanonicalFactError("Base typed node IDs must be unique")
    supplement = (
        supplemental_nodes.select(
            pl.col("node_type").cast(pl.String),
            pl.col("node_id").cast(pl.String),
            pl.col("source_component_path").cast(pl.String),
        )
        .unique(subset=["node_type", "node_id"], keep="first")
        .join(
            base.select("node_type", "node_id"),
            on=["node_type", "node_id"],
            how="anti",
        )
    )
    result = pl.concat([base, supplement]).sort(["node_type", "node_id"])
    return (
        result.with_columns(
            (pl.col("node_id").cum_count().over("node_type") - 1)
            .cast(pl.UInt32)
            .alias("type_local_id")
        )
        .with_columns(
            pl.Series("global_node_id", range(result.height), dtype=pl.UInt32)
        )
        .select(
            "global_node_id",
            "type_local_id",
            "node_type",
            "node_id",
            "source_component_path",
        )
    )


def _source_expr(properties_dtype: pl.DataType, role: str) -> pl.Expr:
    empty = pl.lit([]).cast(pl.List(pl.String))
    if properties_dtype == pl.String:
        return (
            pl.col("properties")
            .str.json_path_match(f"$.sources.{role}")
            .str.json_decode(pl.List(pl.String))
            .fill_null(empty)
        )
    if not isinstance(properties_dtype, pl.Struct):
        return empty
    fields = {field.name: field.dtype for field in properties_dtype.fields}
    sources_dtype = fields.get("sources")
    if not isinstance(sources_dtype, pl.Struct):
        return empty
    source_fields = {field.name for field in sources_dtype.fields}
    if role not in source_fields:
        return empty
    return (
        pl.col("properties")
        .struct.field("sources")
        .struct.field(role)
        .fill_null(empty)
    )


def canonicalize_edge_partition(
    frame: pl.DataFrame,
    node_index: pl.DataFrame,
    spec: EdgeComponentSpec,
    *,
    fact_id_offset: int = 0,
    evidence_id_offset: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create one fact per canonical key and one evidence row per source row."""

    required = {"from", "to", "relation", "undirected", "properties"}
    missing = required - set(frame.columns)
    if missing:
        raise CanonicalFactError(
            f"Edge component {spec.relative_path} lacks {sorted(missing)}"
        )
    if frame.select("from", "to", "relation").null_count().row(0) != (0, 0, 0):
        raise CanonicalFactError(f"Null fact identity in {spec.relative_path}")

    same_type = spec.source_type == spec.target_type
    work = (
        frame.with_columns(
            pl.col("aggregate_source_row_index").alias("source_row_number")
        )
        if "aggregate_source_row_index" in frame.columns
        else frame.with_row_index("source_row_number")
    ).with_columns(
        pl.when(pl.lit(same_type) & pl.col("undirected") & (pl.col("from") > pl.col("to")))
        .then(pl.col("to"))
        .otherwise(pl.col("from"))
        .cast(pl.String)
        .alias("canonical_source_id"),
        pl.when(pl.lit(same_type) & pl.col("undirected") & (pl.col("from") > pl.col("to")))
        .then(pl.col("from"))
        .otherwise(pl.col("to"))
        .cast(pl.String)
        .alias("canonical_target_id"),
    )
    source_map = node_index.filter(pl.col("node_type") == spec.source_type).select(
        pl.col("node_id").alias("canonical_source_id"),
        pl.col("global_node_id").alias("source_global_id"),
    )
    target_map = node_index.filter(pl.col("node_type") == spec.target_type).select(
        pl.col("node_id").alias("canonical_target_id"),
        pl.col("global_node_id").alias("target_global_id"),
    )
    work = (
        work.join(source_map, on="canonical_source_id", how="left", validate="m:1")
        .join(target_map, on="canonical_target_id", how="left", validate="m:1")
        .with_columns(
            pl.concat_str(
                pl.lit(spec.source_type),
                "canonical_source_id",
                "relation",
                pl.lit(spec.target_type),
                "canonical_target_id",
                pl.when(pl.col("undirected"))
                .then(pl.lit("UNDIRECTED"))
                .otherwise(pl.lit("DIRECTED")),
                separator="\u001f",
            ).alias("canonical_fact_key")
        )
    )
    if work.filter(
        pl.col("source_global_id").is_null() | pl.col("target_global_id").is_null()
    ).height:
        raise CanonicalFactError(
            f"Unmapped endpoints in component {spec.relative_path}"
        )

    facts = (
        work.group_by("canonical_fact_key")
        .agg(
            pl.col("canonical_source_id").first().alias("source_node_id"),
            pl.col("canonical_target_id").first().alias("target_node_id"),
            pl.col("source_global_id").first(),
            pl.col("target_global_id").first(),
            pl.col("relation").first(),
            pl.col("undirected").first(),
            pl.len().cast(pl.UInt32).alias("aggregated_observation_count"),
        )
        .sort("canonical_fact_key")
        .with_columns(
            pl.Series(
                "canonical_fact_id",
                range(fact_id_offset, fact_id_offset + work["canonical_fact_key"].n_unique()),
                dtype=pl.UInt64,
            ),
            pl.lit(spec.source_type).alias("source_node_type"),
            pl.lit(spec.target_type).alias("target_node_type"),
            pl.lit(spec.relative_path).alias("source_component_path"),
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
    properties_dtype = frame.schema["properties"]
    evidence = (
        work.join(
            facts.select(
                "canonical_fact_key",
                "canonical_fact_id",
                "aggregated_observation_count",
            ),
            on="canonical_fact_key",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.Series(
                "evidence_record_id",
                range(evidence_id_offset, evidence_id_offset + work.height),
                dtype=pl.UInt64,
            ),
            _source_expr(properties_dtype, "direct").alias("raw_direct_sources"),
            _source_expr(properties_dtype, "indirect").alias("raw_indirect_sources"),
            (pl.col("relation") == "EXPRESSION_PRESENT").alias(
                "edge_support_observed"
            ),
            (pl.col("relation") == "EXPRESSION_ABSENT").alias(
                "edge_counter_evidence_observed"
            ),
            pl.when(
                pl.col("relation").is_in(
                    ["EXPRESSION_PRESENT", "EXPRESSION_ABSENT"]
                )
            )
            .then(pl.lit("GENE_EXPRESSION_PRESENT_IN_ANATOMY"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("edge_evidence_proposition"),
            pl.when(pl.col("aggregated_observation_count") > 1)
            .then(pl.lit(OverlapCategory.UNRESOLVED_OVERLAP.value))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("overlap_category"),
            pl.lit(False).alias("independent_vote_eligible"),
            pl.lit(spec.relative_path).alias("source_component_path"),
        )
        .select(
            "evidence_record_id",
            "canonical_fact_id",
            "canonical_fact_key",
            "source_component_path",
            "source_row_number",
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
        .sort("evidence_record_id")
    )
    return facts, evidence
