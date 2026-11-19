"""Validated configuration for the Phase 4 EviHGT core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EviHGTConfig:
    """Immutable model configuration with fail-fast semantic validation."""

    num_node_types: int
    num_edge_types: int
    num_nodes: int
    hidden_dim: int = 128
    evidence_dim: int = 64
    relation_dim: int = 32
    provenance_dim: int = 16
    num_heads: int = 4
    num_layers: int = 2
    ffn_multiplier: int = 4
    dropout: float = 0.1
    num_numeric_features: int = 0
    num_boolean_features: int = 0
    node_feature_dim: int = 0
    property_token_buckets: int = 8192
    evidence_token_buckets: int = 8192
    provenance_token_buckets: int = 8192
    num_provenance_roles: int = 5
    num_provenance_kinds: int = 6
    num_overlap_categories: int = 5
    use_node_features: bool = False
    use_node_type_embedding: bool = True
    use_node_id_embedding: bool = False
    use_property: bool = True
    use_evidence: bool = True
    use_provenance: bool = True
    use_evidence_context: bool = True
    use_polarity_branches: bool = True
    use_four_valued_logic: bool = True
    use_query_conditioning: bool = True
    return_attention: bool = False
    return_edge_context: bool = False
    return_logic_state: bool = False
    name: str = "EviHGT"

    def __post_init__(self) -> None:
        positive = (
            "num_node_types",
            "num_edge_types",
            "num_nodes",
            "hidden_dim",
            "evidence_dim",
            "relation_dim",
            "provenance_dim",
            "num_heads",
            "num_layers",
            "ffn_multiplier",
            "property_token_buckets",
            "evidence_token_buckets",
            "provenance_token_buckets",
            "num_provenance_roles",
            "num_provenance_kinds",
            "num_overlap_categories",
        )
        for field_name in positive:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "num_numeric_features",
            "num_boolean_features",
            "node_feature_dim",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.use_node_features and self.node_feature_dim <= 0:
            raise ValueError(
                "node_feature_dim must be positive when use_node_features is enabled"
            )
        if not any(
            (
                self.use_node_features,
                self.use_node_type_embedding,
                self.use_node_id_embedding,
            )
        ):
            raise ValueError("at least one node representation source must be enabled")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EviHGTConfig":
        known = {item.name for item in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown EviHGT config fields: {sorted(unknown)}")
        return cls(**dict(value))
