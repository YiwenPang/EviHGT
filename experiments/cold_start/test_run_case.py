import unittest
from pathlib import Path
import json
import os

import pytest

import run_case as cold

from run_case import audit_memberships, tie_diagnostics


def panel(positive, unlabeled, raw_relation):
    return [
        dict(drug_id=drug, disease_id=disease, role=role,
             raw_relation=raw_relation if role == "P" else None)
        for role, pairs in (("P", positive), ("U", unlabeled))
        for drug, disease in pairs
    ]


def _case_fixture(tmp_path):
    import polars as pl

    root = tmp_path / "synthetic case with long name"
    root.mkdir()
    positives = {("held_p", "condition")}
    contra = {("held_q", "condition")}
    unlabeled = {("held_u", "condition")}
    for head, pairs in (("support", positives), ("contra", contra)):
        pl.DataFrame(panel(pairs, unlabeled, cold.RELATIONS[head])).write_parquet(root / cold.PANEL_NAMES[head])
    tables = {"blocked.parquet": positives | contra | unlabeled,
              "observed.parquet": positives | contra,
              "training.parquet": {("seen", "condition")},
              "development.parquet": {("dev", "condition")}}
    for name, pairs in tables.items():
        pl.DataFrame(sorted(pairs), schema=list(cold.PAIR_FIELDS), orient="row").write_parquet(root / name)
    (root / "eligibility.json").write_text(json.dumps({
        "eligible_drugs": ["seen"], "eligible_diseases": ["condition"]}), encoding="utf-8")
    (root / "receipt.json").write_text(json.dumps({
        "case": "cold_drug", "train_updates": 3,
        "test_files": {name: cold._sha(root / name) for name in cold.PANEL_NAMES.values()}}), encoding="utf-8")
    inventory = {name: cold._sha(root / name) for name in (*tables, "eligibility.json", "receipt.json")}
    manifest = {"files": inventory, "panels": {"development": "development.parquet"},
                "eligibility": "eligibility.json", "blocked_pairs": "blocked.parquet",
                "observed_pairs": "observed.parquet", "training_pairs": "training.parquet",
                "training_seed": 42, "schedule": {"G": 1}, "cold_case_receipt": "receipt.json"}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "manifest.json", inventory


def _path_alias(path, spelling, tmp_path, monkeypatch):
    if spelling == "relative":
        monkeypatch.chdir(tmp_path)
        return path.relative_to(tmp_path)
    if spelling == "parent_component":
        (tmp_path / "alias_parent").mkdir()
        return tmp_path / "alias_parent" / ".." / path.relative_to(tmp_path)
    if os.name != "nt":
        pytest.skip("Windows short-name aliases are platform-specific")
    import ctypes
    from ctypes import wintypes
    get_short = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD
    required = get_short(str(path), None, 0)
    if not required:
        pytest.skip("Filesystem does not provide Windows short names")
    buffer = ctypes.create_unicode_buffer(required)
    if not get_short(str(path), buffer, required):
        raise ctypes.WinError(ctypes.get_last_error())
    alias = Path(buffer.value)
    if alias == path:
        pytest.skip("No distinct Windows short-name alias on this filesystem")
    assert alias.resolve(strict=True) == path.resolve(strict=True)
    return alias


@pytest.mark.parametrize("spelling", ["relative", "parent_component", "windows_short"])
def test_preflight_accepts_equivalent_case_paths(tmp_path, monkeypatch, spelling):
    manifest, _ = _case_fixture(tmp_path)
    alias = _path_alias(manifest, spelling, tmp_path, monkeypatch)
    assert cold._preflight("cold_drug", alias) == cold._preflight("cold_drug", manifest.resolve())


@pytest.mark.parametrize("spelling", ["relative", "parent_component", "windows_short"])
def test_content_bound_input_accepts_equivalent_roots(tmp_path, monkeypatch, spelling):
    manifest, inventory = _case_fixture(tmp_path)
    alias = _path_alias(manifest, spelling, tmp_path, monkeypatch)
    assert cold._corp_input(alias, "eligibility.json", inventory) == (manifest.parent / "eligibility.json").resolve()


def test_content_bound_input_still_rejects_escape_and_hash_changes(tmp_path):
    manifest, inventory = _case_fixture(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    for escaped in ("../outside.json", str(outside.resolve())):
        with pytest.raises(ValueError, match="confined relative POSIX"):
            cold._corp_input(manifest, escaped, {escaped: cold._sha(outside)})
    (manifest.parent / "eligibility.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="path or hash"):
        cold._corp_input(manifest, "eligibility.json", inventory)


def test_preflight_still_rejects_panels_outside_the_case(tmp_path, monkeypatch):
    manifest, _ = _case_fixture(tmp_path)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes((manifest.parent / cold.PANEL_NAMES["support"]).read_bytes())
    monkeypatch.setattr(cold, "PANEL_NAMES", {"support": "../outside.parquet", "contra": "test_contra.parquet"})
    with pytest.raises(ValueError, match="stay in their case directory"):
        cold._preflight("cold_drug", manifest)


def test_checkpoint_receipt_normalizes_case_root_and_rejects_escape(tmp_path, monkeypatch):
    root = tmp_path / "case"
    run = root / "frozen_runs/run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/latest.pt").write_bytes(b"synthetic checkpoint identity only")
    (run / "completion.json").write_text(json.dumps({
        "status": "COMPLETE", "variant": "FULL", "mode": "train", "updates": 3, "total_updates": 3}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    receipt = cold._checkpoint_receipt(run.resolve(), Path("case"), {"expected_updates": 3})
    assert receipt["root"] == str(run.resolve())
    outside = tmp_path / "outside_run"
    outside.mkdir()
    with pytest.raises(ValueError, match="result directory"):
        cold._checkpoint_receipt(outside, root, {"expected_updates": 3})


def test_scoring_normalizes_case_root_before_checkpoint_confinement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root = tmp_path / "case"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    checkpoint = root / "latest.pt"
    checkpoint.write_bytes(b"synthetic checkpoint identity only")
    training = {"checkpoint": str(checkpoint), "checkpoint_sha256": cold._sha(checkpoint)}
    monkeypatch.chdir(tmp_path)
    # Stop at the next guard: normalization must not reject a case-local checkpoint.
    with pytest.raises(ValueError, match="manifest has changed"):
        cold._score("cold_drug", None, SimpleNamespace(manifest=str(manifest)), Path("case"),
                    {"manifest_sha256": "deliberately different"}, training)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(checkpoint.read_bytes())
    training["checkpoint"] = str(outside)
    with pytest.raises(ValueError, match="escaped its case"):
        cold._score("cold_drug", None, SimpleNamespace(manifest=str(manifest)), root, {}, training)


class CaseMembershipTests(unittest.TestCase):
    def setUp(self):
        self.positive = {("held_p", "condition")}
        self.contra = {("held_q", "condition")}
        self.unlabeled = {("held_u", "condition")}
        self.blocked = self.positive | self.contra | self.unlabeled
        self.observed = self.positive | self.contra | {("seen", "condition")}
        self.train = {("seen", "condition")}
        self.dev = {("dev", "condition")}
        self.drugs, self.diseases = {"seen"}, {"condition"}
        self.panels = {
            "support": panel(self.positive, self.unlabeled, "INDICATION"),
            "contra": panel(self.contra, self.unlabeled, "CONTRAINDICATION"),
        }

    def audit(self, case="cold_drug"):
        return audit_memberships(case, self.panels, self.blocked, self.observed,
                                 self.train, self.dev, self.drugs, self.diseases)

    def test_certified_cold_drug_rows_can_be_scored(self):
        self.assertEqual(self.audit()["test_union_pairs"], 3)

    def test_eligible_training_endpoint_cannot_enter_drug_cold_test(self):
        # It is not present in the realized positive training pairs, but would be
        # available to the optimizer as a U-sampler endpoint.
        self.drugs.add("held_u")
        with self.assertRaisesRegex(ValueError, "cold axis"):
            self.audit()

    def test_missing_block_fails_before_scoring(self):
        self.blocked.remove(("held_u", "condition"))
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.audit()

    def test_known_pair_cannot_be_test_u(self):
        self.observed.add(("held_u", "condition"))
        with self.assertRaisesRegex(ValueError, "observed"):
            self.audit()

    def test_wrong_relation_cannot_be_test_p(self):
        self.panels["contra"][0]["raw_relation"] = "INDICATION"
        with self.assertRaisesRegex(ValueError, "raw clinical relation"):
            self.audit()

    def test_shared_u_membership_is_pinned(self):
        self.panels["contra"] = panel(self.contra, {("alternate", "condition")}, "CONTRAINDICATION")
        self.blocked.add(("alternate", "condition"))
        with self.assertRaisesRegex(ValueError, "same U"):
            self.audit()

    def test_ties_separate_evaluable_and_unevaluable_diseases(self):
        rows = [
            dict(disease_id="d1", role="P", support_logit=.5),
            dict(disease_id="d1", role="U", support_logit=.5),
            dict(disease_id="d2", role="P", support_logit=.8),
        ]
        stats = tie_diagnostics(rows, "support_logit")
        self.assertEqual(stats["all_diseases"]["constant_score_diseases"], 2)
        self.assertEqual(stats["evaluable_diseases"]["constant_score_diseases"], 1)
        self.assertEqual(stats["evaluable_diseases"]["mean_largest_tie_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
