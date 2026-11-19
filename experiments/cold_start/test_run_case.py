import unittest

from run_case import audit_memberships, tie_diagnostics


def panel(positive, unlabeled, raw_relation):
    return [
        dict(drug_id=drug, disease_id=disease, role=role,
             raw_relation=raw_relation if role == "P" else None)
        for role, pairs in (("P", positive), ("U", unlabeled))
        for drug, disease in pairs
    ]


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
