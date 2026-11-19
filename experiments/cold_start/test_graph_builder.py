import numpy as np
import polars as pl

from graph_builder import filter_fact_identities, mask_csr


def test_removes_incidence_and_both_directions_of_blocked_drug_disease():
    facts = pl.DataFrame(
        {
            "canonical_fact_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "source_node_type": ["DRG", "GEN", "DIS", "DRG", "DIS", "GEN", "DRG", "GEN"],
            "target_node_type": ["GEN", "DRG", "GEN", "DIS", "DRG", "GEN", "DIS", "DRG"],
            "source_node_id": ["D0", "G1", "Z0", "D1", "Z1", "G1", "D2", "G2"],
            "target_node_id": ["G1", "D0", "G2", "Z1", "D1", "G2", "Z2", "D1"],
        }
    )
    retained = filter_fact_identities(facts.lazy(), {"D0"}, {"Z0"}, {("D1", "Z1")}, {6}).collect()
    assert retained["canonical_fact_id"].to_list() == [7, 8]


def test_removes_both_csr_traversals_and_preserves_neighbor_order():
    csr = {
        "indptr": np.array([0, 2, 4, 6], dtype=np.int64),
        "indices": np.array([1, 2, 0, 2, 0, 1], dtype=np.uint32),
        "fact_ids": np.array([11, 12, 11, 13, 12, 13], dtype=np.uint64),
    }
    kept = mask_csr(csr, np.array([12], dtype=np.uint64), np.array([1], dtype=np.uint32))
    assert kept["indptr"].tolist() == [0, 1, 1, 2]
    assert kept["indices"].tolist() == [2, 0]
    assert kept["fact_ids"].tolist() == [12, 12]
