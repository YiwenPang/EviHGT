"""Canonical traversal contract and clinical target relation names."""
from dataclasses import dataclass

DDA_CLINICAL_RELATIONS = frozenset({"INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE"})

@dataclass(frozen=True, slots=True)
class ExtractedSubgraph:
    node_ids: tuple[int, ...]
    fact_ids: tuple[int, ...]
    traversal_edges: tuple[tuple[int, int, int], ...]
