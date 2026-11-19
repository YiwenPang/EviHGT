"""Invented CPU fixture: exercise scoring only after all three tiny training updates."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_case as cold


def smoke():
    runner, settings_module = cold._bootstrap()
    original_source = cold.FROZEN / "tests"
    sys.path.insert(0, str(original_source))
    from data_fixture import make_corpus
    import numpy as np
    import polars as pl
    import torch

    torch.set_num_threads(1)
    root = Path(tempfile.mkdtemp(prefix="SYNTHETIC_TEST_ONLY_"))
    try:
        case = "cold_disease"
        corpus_path = make_corpus(root / "corpus")
        corpus = corpus_path.parent
        m = cold._read_json(corpus_path)
        nodes_path = corpus / m["nodes"]
        nodes = pl.read_parquet(nodes_path)
        added_id = nodes.height
        new_row = pl.DataFrame({"global_node_id": [added_id], "node_id": ["S3"], "node_type": ["DIS"]})
        pl.concat([nodes, new_row]).write_parquet(nodes_path)
        ptr = corpus / m["csr"]["indptr"]
        indptr = np.load(ptr)
        np.save(ptr, np.r_[indptr, indptr[-1]], allow_pickle=False)
        for key, row in (("disease_ids", [added_id]), ("disease_values", None)):
            target = corpus / m["raw_features"][key]
            a = np.load(target)
            patched = np.r_[a, row] if row else np.concatenate([a, np.zeros_like(a[:1])])
            np.save(target, patched, allow_pickle=False)
        for key in ("disease_values", "disease_available"):
            target = corpus / m["pretrained"][key]
            a = np.load(target)
            np.save(target, np.concatenate([a, np.zeros_like(a[:1])]), allow_pickle=False)
        for key, drugs in (("blocked_pairs", ["D0", "D1", "D2"]), ("observed_pairs", ["D0", "D1"])):
            target = corpus / m[key]
            append = pl.DataFrame({"drug_id": drugs, "disease_id": ["S3"] * len(drugs)})
            pl.concat([pl.read_parquet(target), append]).write_parquet(target)
        for head, drug in (("support", "D0"), ("contra", "D1")):
            panel = pl.DataFrame({"drug_id": [drug, "D2"], "disease_id": ["S3", "S3"],
                                  "role": ["P", "U"], "raw_relation": [cold.RELATIONS[head], None]})
            panel.write_parquet(corpus / cold.PANEL_NAMES[head])
        m["cold_case_receipt"] = "preparation_audit.json"
        cold._write_json(corpus / m["cold_case_receipt"],
                         {"case": case, "train_updates": 3, "test_files":
                           {name: cold._sha(corpus / name) for name in cold.PANEL_NAMES.values()}})
        m["files"][m["cold_case_receipt"]] = ""
        m["files"] = {name: cold._sha(corpus / name) for name in m["files"]}
        cold._write_json(corpus_path, m)
        settings = replace(cold._training_settings(settings_module, corpus_path, root / "frozen_runs"),
                           device="cpu", precision="fp32")
        audit = cold._preflight(case, corpus_path)
        assert audit["test_union_pairs"] == 3
        runner.Console = cold._localized_console(runner.Console)
        path = runner.run(settings, variant="FULL", mode="train")
        receipt = cold._checkpoint_receipt(path, root, audit)
        result = cold._score(case, runner, settings, root, audit, receipt)
        assert result["status"] == "COMPLETE"
        assert result["updates"] == 3
        assert result["graph_coverage"]["distinct_pairs"] == 3
        assert result["graph_coverage"]["disease_endpoint_with_fact"] == 0
        assert result["graph_coverage"]["pairs_with_path"] == 0
        for head, values in result["metrics"].items():
            assert values["evaluable_diseases"] == 1, head
            assert values["macro_auc"] is not None, head
            assert len(values["ranking"]) == 8, head
            assert result["tie_diagnostics"][head]["evaluable_diseases"]["diseases"] == 1
        assert cold._read_json(root / "test/metrics.json") == result
        print(f"SYNTHETIC TEST ONLY: CPU TRAIN/FINAL-TEST SMOKE PASSED: {root}")
    except BaseException:
        print(f"SYNTHETIC TEST ONLY: CPU SMOKE FAILED: {root}", file=sys.stderr)
        raise


if __name__ == "__main__":
    smoke()
