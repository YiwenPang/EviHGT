import unittest

from prepare_case import entity_partition, make_panels, make_schedule


class ColdCaseTests(unittest.TestCase):
    def test_panel_candidates_never_draw_observed_or_protected_pairs(self):
        sup = {('a', 'd')}
        con = {('b', 'other')}
        observed = sup | con | {('b', 'd')}
        protected = {('c', 'd'), ('c', 'other')}
        support, contra, unlabeled = make_panels(sup, con, {'a', 'b', 'c', 'u'}, observed | protected, 'test')
        self.assertEqual(unlabeled, {('u', 'd'), ('a', 'other'), ('u', 'other')})
        self.assertEqual(set(support.filter(role='P').select('drug_id','disease_id').iter_rows()), sup)
        self.assertEqual(set(contra.filter(role='P').select('drug_id','disease_id').iter_rows()), con)

    def test_3g_positive_coverage_and_only_legal_u(self):
        sup = {('a', 'x'), ('a', 'y'), ('b', 'y')}
        con = {('b', 'x'), ('c', 'x')}
        excluded = sup | con | {('u', 'y')}
        G, rows, records = make_schedule(sup, con, {'a', 'b', 'c', 'u'}, {'x','y'}, excluded, 'test')
        self.assertEqual((G, len(records)), (2, 4))
        for cycle in (1, 2, 3):
            sub = rows.filter(cycle=cycle)
            for role, expected in [('support', sup), ('unsuitability', con)]:
                pairs = sub.filter(role=role).select('drug_id','disease_id').rows()
                self.assertEqual(len(pairs), len(expected))
                self.assertEqual(set(pairs), expected)
            self.assertFalse(excluded & set(sub.filter(role='U').select('drug_id','disease_id').rows()))
        for row in records:
            self.assertFalse(excluded & {tuple(p) for p in row['extra_pairs']})

    def test_hash_assignment_is_repeatable_and_has_all_partitions(self):
        ids = [str(x) for x in range(1000)]
        one = {i: entity_partition(i, 'drug_id') for i in ids}
        self.assertEqual(one, {i: entity_partition(i, 'drug_id') for i in reversed(ids)})
        self.assertEqual(set(one.values()), {'train','validation','test'})


if __name__ == '__main__':
    unittest.main()
