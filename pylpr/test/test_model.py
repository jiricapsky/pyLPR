from argparse import Namespace
import unittest
from pylpr.model import LPR_model
import pulp as pl
import os
import numpy as np

from pylpr.data import Graph_names

class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        args = Namespace(
            rules_file = 'pylpr/test/test_rules.npy',
            solver = 'PULP_CBC_CMD',
            iterations = 20,
            rules_file_temp = 'pylpr/test/test_rules.npy',
            rules_load = True,
            skip_writing = True,
            skip_neg = True,
            skip_weight = True,
            cores = 1,
            seed = 12345,
            max_length = 4,
            column_generation=False
        )
        tradeoff = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
        self.model= LPR_model("pylpr/test/testdataset", tradeoff, args)

    def test_get_rules_h1(self):
        # all rules with length 1 or 2
        # output are relational paths for each fact from head to tail entity
        # inital fact not needed in result, because eveything is abstracted to relations
        # -> always starts with the relation from argument
        # e1 r2 e4 : ((e1 inv_r1 e4)) -> output: (inv_r1)
        ## GRAPH WITH INVERSE RELATIONS
        # 1 -> r2
        inv_r1 = self.model.data.rel_to_inv(0)
        rules = self.model._get_rules_h1(Graph_names.Train, 1)
        expected = {
            (0,),
            (inv_r1, 0),
            (inv_r1,),
            (0, 0)
        }
        self.assertSetEqual(rules, expected)

    def test_bfs(self):
        # for single fact get shortest path and shortest + 1
        ## GRAPH WITH INVERSE RELATIONS
        # fact: e1 r1 e4
        inv_r1 = self.model.data.rel_to_inv(0)
        inv_r2 = self.model.data.rel_to_inv(1)
        paths = self.model.bfs(Graph_names.Train, (0, 0, 3), True)
        expected = [
            (inv_r2,),
            (inv_r1, inv_r1)
        ]
        self.assertListEqual(paths, expected)

        # fact: e2 r1 e3
        paths = self.model.bfs(Graph_names.Train, (1, 0, 2), True)
        expected = [
            (0, inv_r1) ,
            (inv_r1, 1, inv_r1)
        ]
        self.assertListEqual(paths, expected)

    def test_rules_h2(self):
        # for each fact find shortest path and shortest + 1
        ## GRAPH WITH INVERSE RELATIONS
        # 0 -> r1
        # 1 -> r2
        inv_r1 = self.model.data.rel_to_inv(0)
        inv_r2 = self.model.data.rel_to_inv(1)
        rules = self.model.get_rules_h2(Graph_names.Train, 0)
        expected = {
            # e2 r1 e3
            (0, inv_r1),
            (inv_r1, 1, inv_r1),
            # e2 e1 e1
            (0, 0),
            # e3 r1 e1
            (1,),
            (inv_r1, 0),
            # e1 r1 e4
            (inv_r2,),
            (inv_r1, inv_r1),
            # e4 r1 e2
            (1, inv_r1),
            (1, inv_r1, inv_r1)
        }
        self.assertSetEqual(rules, expected)

    def test_can_append_to_path_new(self):
        # method doesn't check head entity -> only used in bfs method and head is already checked
        # same fact
        # first item always the original fact
        path = [(0, 0, 1),]
        fact = (0, 0, 1)
        self.assertFalse(self.model.can_append_to_path(path, fact))

        # len 1 - valid
        path = [(0, 0, 1)]
        fact = (0, 1, 2)
        self.assertTrue(self.model.can_append_to_path(path, fact))

        # len 2 - valid, visited entity is tail of original fact
        path = [(0, 0, 1), (0, 1, 2)]
        fact = (2, 1, 1)
        self.assertTrue(self.model.can_append_to_path(path, fact))

        # same head as start, len 2
        path = [(0, 0, 1), (0, 1, 2)]
        fact = (2, 1, 0)
        self.assertFalse(self.model.can_append_to_path(path, fact))

    def test_all_rules(self):
        # test all generated rules using h1 and h2
        # not really needed but next tests are using these generated rules
        inv_r1 = self.model.data.rel_to_inv(0)
        inv_r2 = self.model.data.rel_to_inv(1)
        rules_h1 = self.model._get_rules_h1(Graph_names.Train, 1)
        rules_h2 = self.model.get_rules_h2(Graph_names.Train, 1)
        rules = rules_h1.union(rules_h2)
        expected_h1 = {
            (0,),
            (inv_r1, 0),
            (inv_r1,),
            (0, 0)
        }
        expected_h2 = {
            # e4 r2 e1
            (inv_r1,),
            (0, 0),
            # e3 r2 e1
            (0,),
            (inv_r1, 0)
        }
        expected = {
            (inv_r1,),
            (0,),
            (0, 0),
            (inv_r1, 0),
        }

        self.assertSetEqual(rules, expected)

    def test_write_load_rules(self):
        inv_r1 = self.model.data.rel_to_inv(0)
        rules = np.array(
            [
                self.model.create_rule(1, (inv_r1,)),
                self.model.create_rule(1, (0,)),
                self.model.create_rule(1, (0, 0)),
                self.model.create_rule(1, (0, inv_r1))
            ],
            dtype = self.model.dtype_rule
        )
        if os.path.exists(self.model.file_rules):
            os.remove(self.model.file_rules)
        self.model.write_rules(self.model.file_rules, [rules,])

        loaded_rules = self.model.load_rules(self.model.file_rules)

        self.assertEqual(len(rules), len(loaded_rules))
        for rule in loaded_rules:
            self.assertIn(rule, rules)

    def test_calculate_neg_freq(self):
        # rule: 1 -> 0
        rule = np.array([self.model.create_rule(1, (0,)),], dtype=self.model.dtype_rule)[0]
        neg, freq = self.model.calculate_neg_freq(Graph_names.Train, rule)

        expected_neg = 1 + 1
        expected_freq = 1 + 1 + expected_neg

        self.assertEqual(neg, expected_neg)
        self.assertEqual(freq, expected_freq)

        # rule: 1 -> 0, inv_r1
        rule = np.array([self.model.create_rule(1, (0, self.model.data.rel_to_inv(0))),], dtype=self.model.dtype_rule)[0]
        neg, freq = self.model.calculate_neg_freq(Graph_names.Train, rule)

        expected_neg = 1 + 0
        expected_freq = 0 + 0 + expected_neg

        self.assertEqual(neg, expected_neg)
        self.assertEqual(freq, expected_freq)

    def test_find_endpoints_from_node(self):
        rel_path = (0,)
        endpoints = self.model.find_endpoints_from_node(Graph_names.Train, 0, rel_path)
        expected = [3,]
        for e in expected:
            self.assertIn(e, endpoints)

        rel_path = (0, self.model.data.rel_to_inv(0))
        endpoints = self.model.find_endpoints_from_node(Graph_names.Train, 2, rel_path)
        expected = [1,]
        for e in expected:
            self.assertIn(e, endpoints)

        rel_path = (0, self.model.data.rel_to_inv(0))
        endpoints = self.model.find_endpoints_from_node(Graph_names.Train, 0, rel_path)
        expected = []
        for e in expected:
            self.assertIn(e, endpoints)

    def test_init_base_problem(self):
        self.model.rules = self.model.load_rules(self.model.args.rules_file)
        problem, weights, penalty = self.model._init_base_problem(1, Graph_names.Train)
        # penalty for each edge
        expected_penalty = [
            pl.LpVariable("Penalty_3:0", lowBound=0.0, upBound=1.0),
            pl.LpVariable("Penalty_2:0", lowBound=0.0, upBound=1.0)
        ]

        for p in penalty:
            self.assertIn(p, expected_penalty)

        # weights
        inv_r1 = self.model.data.rel_to_inv(0)
        rules = np.array([
            self.model.create_rule(1, (0,)),
            self.model.create_rule(1, (0, 0)),
            self.model.create_rule(1, (inv_r1,)),
            self.model.create_rule(1, (inv_r1, 0))
        ],
        dtype=self.model.dtype_rule)

        weights_expected = [
            (pl.LpVariable("w_0", 0.0, 1.0), rules[0]),
            (pl.LpVariable("w_1", 0.0, 1.0), rules[1]),
            (pl.LpVariable("w_2", 0.0, 1.0), rules[2]),
            (pl.LpVariable("w_3", 0.0, 1.0), rules[3])
        ]

        # constraints for edge
        expected_constraint = {
            'c_2:0': (pl.lpSum([weights[0][0], weights[3][0]]) + expected_penalty[1]) >= 1.0,
            'c_3:0': (pl.lpSum([weights[1][0], weights[2][0]]) + expected_penalty[0]) >= 1.0,
        }

        self.assertDictEqual(problem.constraints, expected_constraint)

    def test_path_exists(self):
        # 1 -> 0
        rule = np.array([self.model.create_rule(1, (0,)),], dtype=self.model.dtype_rule)[0]
        rule_antecedent = self.model.get_rule_antecedent(rule)
        path_exist = self.model.path_exists(rule_antecedent, (3, 0), Graph_names.Train)
        self.assertFalse(path_exist)

        path_rev = self.model.reverse_path(rule_antecedent)
        path_exist = self.model.path_exists(path_rev, (3, 0), Graph_names.Train)
        self.assertTrue(path_exist)

        path_exist = self.model.path_exists(rule_antecedent, (2, 0), Graph_names.Train)
        self.assertTrue(path_exist)

        path_rev = self.model.reverse_path(rule_antecedent)
        path_exist = self.model.path_exists(path_rev, (0, 2), Graph_names.Train)
        self.assertTrue(path_exist)

        # 1 -> 0, r_0
        inv_r1 = self.model.data.rel_to_inv(0)
        rule = np.array([self.model.create_rule(1, (0, inv_r1)),], dtype=self.model.dtype_rule)[0]
        rule_antecedent = self.model.get_rule_antecedent(rule)
        path_exist = self.model.path_exists(rule_antecedent, (0, 2), Graph_names.Train)
        self.assertFalse(path_exist)

        path_exist = self.model.path_exists(rule_antecedent, (1, 3), Graph_names.Train)
        self.assertFalse(path_exist)

    def test_entity_score(self):
        # fact (3, 1, 0)
        inv_r1 = self.model.data.rel_to_inv(0)

        rules = np.array([
            self.model.create_rule(1, (0,), weight=1.0),
            self.model.create_rule(1, (0, 0), weight=1.0),
            self.model.create_rule(1, (inv_r1,), weight=1.0),
            self.model.create_rule(1, (inv_r1, 0), weight=1.0)
        ],
        dtype=self.model.dtype_rule)
        weights_paths = [(r['weight'], self.model.get_rule_antecedent(r)) for r in rules]

        score = self.model.entity_score(Graph_names.Train, weights_paths, (3, 0))
        expected_score = 2 # rules: 1, 2
        self.assertEqual(score, expected_score)

        # alternative head entities from next test
        score = self.model.entity_score(Graph_names.Train, weights_paths, (3, 1))
        expected_score = 1
        self.assertEqual(score, expected_score)

        score = self.model.entity_score(Graph_names.Train, weights_paths, (3, 2))
        expected_score = 1
        self.assertEqual(score, expected_score)

        score = self.model.entity_score(Graph_names.Train, weights_paths, (3, 4))
        expected_score = 0
        self.assertEqual(score, expected_score)


    def test_get_rank(self):
        # rank for tail in fact (3, 1, 0)
        fact = (3,1,0)
        inv_r1 = self.model.data.rel_to_inv(0)

        rules = np.array([
            self.model.create_rule(1, (0,), weight=1.0),
            self.model.create_rule(1, (0, 0), weight=1.0),
            self.model.create_rule(1, (inv_r1,), weight=1.0),
            self.model.create_rule(1, (inv_r1, 0), weight=1.0)
        ],
        dtype=self.model.dtype_rule)

        weights_paths = [(r['weight'], self.model.get_rule_antecedent(r)) for r in rules]
        weights_paths_rev = []
        for weight, path in weights_paths:
            weights_paths_rev.append((weight, self.model.reverse_path(path)))

        entities = self.model.data.get_heads_corrupted_for_rel_tail(Graph_names.Train, self.model.data.rel_to_inv(fact[1]), fact[2])
        scores = [(
                entity,
                self.model.entity_score(Graph_names.Train, weights_paths_rev, (fact[2], entity)))
            for entity in entities
        ]
        true_score = self.model.entity_score(Graph_names.Train, weights_paths_rev, (fact[2], fact[0]))

        # (x, r, h)
        scores_expected = [
            (0, 0),
            (1, 2),
            (4, 0),
            (5, 0)
        ]
        true_expected = 2
        rank = self.model.get_rank(true_score, [score for _, score in scores])

        self.assertListEqual(scores, scores_expected)
        self.assertIn(rank, range(1, 3))

        # rank for head in fact (3, 1, 0)
        entities = self.model.data.get_heads_corrupted_for_rel_tail(Graph_names.Train, fact[1], fact[0])
        scores = [(
                entity,
                self.model.entity_score(Graph_names.Train, weights_paths, (fact[0], entity)))
            for entity in entities
        ]
        true_score = self.model.entity_score(Graph_names.Train, weights_paths, (fact[0], fact[2]))

        scores.sort(reverse=True, key=lambda entity_score: entity_score[1])

        # (r, r, x)
        scores_expected = [
            (1, 1),
            (2, 1),
            (3, 0),
            (4, 0),
            (5, 0)
        ]
        true_expected = 2

        rank = self.model.get_rank(true_score, [score for _, score in scores])

        self.assertListEqual(scores, scores_expected)
        self.assertEqual(rank, 1)

if __name__ == '__main__':
    unittest.main()
