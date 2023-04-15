import unittest
from pylpr.data import Data, Graph_names

class TestData(unittest.TestCase):
    def setUp(self) -> None:
        self.data = Data("pylpr/test/testdataset")

    def test_entity_count_new(self):
        self.assertEqual(self.data.num_entity, 6)

    def test_relation_count_new(self):
        self.assertEqual(self.data.num_relation, 4)

    def test_graphs_created_new(self):
        # Test if all 3 graphs have been created
        self.assertEqual(len(self.data.graphs), 3)
        self.assertTrue(Graph_names.Train in self.data.graphs.keys())
        self.assertTrue(Graph_names.Test in self.data.graphs.keys())
        self.assertTrue(Graph_names.Validate in self.data.graphs.keys())

    def test_is_inverse_new(self):
        for rel in self.data.relation_to_num.values():
            self.assertFalse(self.data.is_inverse(rel))

        # negative value
        self.assertFalse(self.data.is_inverse(-1))

    def test_rel_to_inv_new(self):
        # Converting normal relations to inverse
        rel_count = self.data.num_relation
        # existing relations
        for rel in self.data.relation_to_num.values():
            # each inverse rel is offset by by rel_count and is not one of existing relations
            self.assertEqual(self.data.rel_to_inv(rel), rel + rel_count)
            self.assertFalse(self.data.rel_to_inv(rel) in self.data.relation_to_num.values())

            # already inverse to inverse -> remains inverse
            inv = self.data.rel_to_inv(rel)
            self.assertEqual(self.data.rel_to_inv(inv), self.data.rel_to_inv(rel))

        # really high value
        high_val = 10 * rel_count
        self.assertEqual(self.data.rel_to_inv(high_val), high_val)

        # really low value
        low_val = -10 * rel_count
        self.assertEqual(self.data.rel_to_inv(low_val), low_val)

    def test_inv_to_rel_new(self):
        # Converting inverse relations to normal
        rel_count = self.data.num_relation
        # existing relations
        for rel in self.data.relation_to_num.values():
            inv = self.data.rel_to_inv(rel)
            self.assertEqual(self.data.inv_to_rel(inv), rel)

            # normal to normal
            self.assertEqual(self.data.inv_to_rel(rel), rel)

            # too high value
            too_high = rel + rel_count * 10
            self.assertLessEqual(self.data.inv_to_rel(too_high), rel_count)

        # really low value
        low_val = -10 * rel_count
        self.assertEqual(self.data.inv_to_rel(low_val), 0)

    def test_get_tails_for_rel_head_new(self):
        # Get next entities for specified entity and relation
        ## GRAPH WITH INVERSE RELATIONS
        ## normal
        # 0 -> r1
        # 0 -> e1
        expected_result = [2, 1]
        self.assertCountEqual(self.data.get_tails_for_rel_entity(Graph_names.Train, 0, 0, False), expected_result)
        # reverse
        expected_result = [3]
        self.assertCountEqual(self.data.get_tails_for_rel_entity(Graph_names.Test, 0, 0, True), expected_result)

        ## inverse
        # 0 -> r1 -> inverse
        # 1 -> e2
        inv_r1 = self.data.rel_to_inv(0)
        expected_result = [0, 2]
        self.assertCountEqual(self.data.get_tails_for_rel_entity(Graph_names.Train, inv_r1, 1, False), expected_result)
        # reverse
        expected_result = [3]
        self.assertCountEqual(self.data.get_tails_for_rel_entity(Graph_names.Train, inv_r1, 1, True), expected_result)

        ## GRAPH WITHOUT INVERSE RELATIONS
        ## inverse
        # 1 -> r2 -> inverse
        # 3 -> e4
        inv_r2 = self.data.rel_to_inv(1)
        expected_result = [0]
        self.assertCountEqual(self.data.get_tails_for_rel_entity(Graph_names.Test, inv_r2, 3, False), expected_result)
        # reverse
        # 1 -> r2 -> inverse
        # 0 -> e1
        expected_result = [3, 2]
        self.assertCountEqual(self.data.get_tails_for_rel_entity(Graph_names.Test, inv_r2, 0, True), expected_result)

    def test_get_facts_for_rel_new(self):
        # Get facts for single relation as a dict[int,[int]]
        # GRAPH WITH INVERSE RELATIONS
        # 0 -> r1
        facts = {}
        for h, t in self.data.get_facts_for_rel(Graph_names.Train, 0):
            facts[h] = t
        expected = {
            0: [1, 2],
            1: [3],
            2: [1],
            3: [0]
        }
        self.assertEqual(len(facts.keys()), len(expected.keys()))
        self.assertCountEqual(facts[0], expected[0])
        self.assertCountEqual(facts[1], expected[1])
        self.assertCountEqual(facts[2], expected[2])
        self.assertCountEqual(facts[3], expected[3])

        ## inverse
        # 0 -> r1 -> inverse
        inv_r1 = self.data.rel_to_inv(0)
        facts = {}
        for h ,t in self.data.get_facts_for_rel(Graph_names.Train, inv_r1):
            facts[h] = t
        expected = {
            0: [3],
            1: [2, 0],
            2: [0],
            3: [1]
        }
        self.assertEqual(len(facts.keys()), len(expected.keys()))
        self.assertCountEqual(facts[0], expected[0])
        self.assertCountEqual(facts[1], expected[1])
        self.assertCountEqual(facts[2], expected[2])
        self.assertCountEqual(facts[3], expected[3])

    def test_get_tails_new(self):
        # get next entities for all relations
        ## GRAPH WITH INVERSE RELATIONS
        # 0 -> e1
        # 0 -> r1
        # 1 -> r2
        # 3 -> inv_r1
        inv_r1 = self.data.rel_to_inv(0)
        result = {}
        for r, entities in self.data.get_tails(Graph_names.Train, 0):
            result[r] = entities
        expected = {
            0: [2, 1],
            1: [2, 3],
            inv_r1: [3]
        }
        self.assertEqual(len(result), len(expected))
        self.assertCountEqual(result[0], expected[0])
        self.assertCountEqual(result[1], expected[1])
        self.assertCountEqual(result[inv_r1], expected[inv_r1])

if __name__ == "__main":
    unittest.main()
