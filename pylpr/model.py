import os
import re

from numpy._typing import NDArray
from .data import Data, Graph_names
import pulp as pl
from multiprocessing import Pool
import numpy as np

DEFAULT_SOLVER = 'PULP_CBC_CMD'


class LPR_model:
    def __init__(self, dataset_dir, tradeoff, args) -> None:
        self.data = Data(dataset_dir)
        self.tradeoff = tradeoff

        self.args = args
        self.file_rules = args.rules_file
        self.file_rules_temp = args.rules_file_temp
        self.max_rule_length = args.max_length
        self.skip_writing = args.skip_writing
        self.skip_neg = args.skip_neg
        self.skip_weight = args.skip_weight
        self.cores = args.cores
        self.iterations = args.iterations
        self.seed = args.seed
        self.use_column_generation = args.column_generation
        self.rules_load = args.rules_load

        self.rng = np.random.default_rng(self.seed)

        self.dtype_fact = np.dtype([('h', np.uint16), ('r', np.uint16), ('t', np.uint16)])
        self.dtype_rule = np.dtype([
            ('consequent', np.uint16),
            ('antecedent', np.uint16, (int(self.max_rule_length),)),
            ('weight', float),
            ('freq', np.uint16),
            ('neg', np.uint16),
            ('length', np.uint8)
        ])

        self.rules = np.array([], dtype=self.dtype_rule)

    def fit(self):
        # get rules
        if self.rules_load:
            if not os.path.isfile(self.file_rules_temp):
                print(f'Rules couln\'t be loaded, file \'{self.file_rules_temp}\' not found')
                return
            print(f'Loading rules from file \'{self.file_rules_temp}\'')
            self.rules = self.load_rules(self.file_rules_temp)

        else:
            print('Generating rules for relations:')
            rules = []
            # column generation
            if self.use_column_generation:
                solver = self.get_solver()
                with Pool(processes=self.cores) as pool:
                    rules = pool.starmap(self.get_rules_for_rel_column,
                                         [[rel, Graph_names.Train, solver.copy()] for rel in
                                          self.data.relation_to_num.values()])

            # normal
            else:
                with Pool(processes=self.cores) as pool:
                    rules = pool.starmap(self.generate_rules_for_rel,
                                         [[Graph_names.Train, rel] for rel in self.data.relation_to_num.values()])

            self.rules = np.concatenate([r for r in rules if not None])

            if not self.skip_writing:
                print(f'Writing rules to file \'{self.file_rules_temp}\'')
                self.write_rules(self.file_rules_temp, rules)

        self.print_rules_info(self.rules, True)

        # calculate neg
        if not self.skip_neg | self.use_column_generation:
            print('Calculating neg for rules:')
            rules = []
            with Pool(processes=self.cores) as pool:
                rules = pool.map(self.get_rules_with_updated_neg_freq,
                                 np.unique(self.rules['consequent']))

            self.rules = np.concatenate([r for r in rules])

            if not self.skip_writing:
                print(f'Writing rules to file \'{self.file_rules_temp}\'')
                self.write_rules(self.file_rules_temp, rules)

        # Calculate weight
        if not self.skip_weight:
            print('Calculating weight for rules:')
            solver = self.get_solver()
            params = [[rel, solver.copy()] for rel in np.unique(self.rules['consequent'])]

            rules = []
            with Pool(processes=self.cores) as pool:
                rules = pool.starmap(self.solve_for_rel, params)

            self.rules = np.concatenate([r for r in rules])
            if not self.skip_writing:
                print(f'Writing rules to file \'rules_updated.npy\'')
                self.write_rules(self.file_rules, rules)

    def predict(self):
        print('Evaluating on test dataset')
        if self.rules_load:
            self.rules = self.load_rules(self.file_rules)
        result = {}
        mrr_ranks = []
        with Pool(processes=self.cores) as pool:
            mrr_ranks = pool.starmap(self.get_mrr_and_ranks,
                                     [[rel, Graph_names.Test] for rel in self.data.relation_to_num.values()])

        all_ranks = []
        for rel, mrr, ranks in mrr_ranks:
            hits_1 = self.hits_k(ranks, 1)
            hits_3 = self.hits_k(ranks, 3)
            hits_10 = self.hits_k(ranks, 10)
            result[rel] = {'mrr': mrr, 'hits_1': hits_1, 'hits_3': hits_3, 'hits_10': hits_10}
            all_ranks.extend(ranks)
        return result, all_ranks

    #
    # LOADDING / WRITING
    #
    def write_rules(self, filepath, rules: list[NDArray]):
        np.save(filepath, np.concatenate([r for r in rules], axis=0))

    def load_rules(self, filepath) -> NDArray:
        rules = np.load(filepath)
        return rules

    #
    # HEURISTICS
    #
    def _get_rules_h1(self, graph_name: Graph_names, rel: int) -> set[tuple[int]]:
        rel_paths = set()
        for tail, heads in self.data.get_facts_for_rel(graph_name, rel):
            for head in heads:
                paths = []
                fact = (tail, rel, head)
                for r, entities_for_r in self.data.get_heads(graph_name, tail):
                    for e in entities_for_r:
                        found_fact = (tail, r, e)
                        if found_fact == fact:
                            continue

                        # completes rel_path
                        if e == head:
                            rel_paths.add((r,))

                        paths.append((r, e))
                for prev_relation, prev_entity in paths:
                    for r, next_entities in self.data.get_heads(graph_name, prev_entity):
                        for e in next_entities:
                            found_fact = (prev_entity, r, e)
                            if found_fact == fact:
                                continue

                            # completes rel_path
                            if e == head:
                                rel_paths.add((prev_relation, r))

        return rel_paths

    def get_rules_h2(self, graph_name: Graph_names, rel: int):
        rules = set()
        for tail, heads in self.data.get_facts_for_rel(graph_name, rel):
            for head in heads:
                paths = self.bfs(graph_name, (tail, rel, head),
                                 True, self.max_rule_length)
                rules.update(paths)
        return rules

    def get_rules_for_rel_column(self, rel, graph, solver, iterations=15, max_per_iter=10):
        min_t = min(self.tradeoff)
        min_k = 4
        found_paths = set()

        problem, weights_rules, constraints = self.init_custom_problem(rel, [], graph, min_t, min_k)
        problem.solve(solver)
        dual_complex, dual_edges = self.get_dual_vars(problem.constraints.copy())
        last = 0

        for i in range(1, iterations + 1):
            # find new rules
            edges = ((edge, dual_edges[edge]) for edge in sorted(dual_edges, key=dual_edges.get, reverse=True) if
                     dual_edges[edge] > 0)

            # paths
            paths = set()
            for edge, dual in edges:
                path = self.bfs(graph, (edge[0], rel, edge[1]), False, self.max_rule_length)
                if len(path) == 0:
                    continue
                if path[0] in found_paths:
                    continue
                paths.add(path[0])
                if len(paths) >= max_per_iter:
                    break

            found_paths = found_paths.union(paths)
            new_rules = np.array([self.create_rule(rel, path) for path in paths], dtype=self.dtype_rule)
            for rule in new_rules:
                neg, freq = self.calculate_neg_freq(graph, rule)
                rule['neg'] = neg
                rule['freq'] = freq

            # red_k
            rules_selected = [rule for rule in new_rules if
                              self.get_red_k(rule, min_t, dual_edges, dual_complex, graph) < 0]
            if len(rules_selected) == 0:
                break

            rules_all = [rule for _, rule in weights_rules] + rules_selected

            problem, weights_rules, constraints = self.init_custom_problem(rel, rules_all, graph, min_t, min_k)
            problem.solve(solver)
            dual_complex, dual_edges = self.get_dual_vars(problem.constraints.copy())

        print(f'- relation \'{rel}\': done')

        if len(weights_rules) == 0:
            return None

        return [r for _, r in weights_rules]

    #
    # GRAPH TRAVERSAL
    #
    def bfs(self, graph_name: Graph_names, fact: tuple[int, int, int], include_longer=True, max_length=5):
        path_shortest = None
        path_longer = None
        result = []
        paths_new = [[fact], ]
        paths_old = []

        inverse_included = self.data.is_inverse_included(graph_name)

        for _ in range(max_length):
            paths_old = paths_new.copy()
            paths_new = []

            for path in paths_old:
                last_entity = path[-1][0] if len(path) == 1 else path[-1][2]
                for r, next_entities in self.data.get_heads(graph_name, last_entity):
                    for entity in next_entities:
                        if not self.can_append_to_path(path, (last_entity, r, entity)):
                            continue

                        valid_path = path.copy()
                        valid_path.append((last_entity, r, entity))

                        # fact completes path
                        if entity == fact[2]:
                            # shortest not found
                            if path_shortest is None:
                                path_shortest = valid_path[1::]
                                result.append(tuple([f[1] for f in path_shortest]))
                                if not include_longer:
                                    return result
                                continue
                            # shortest found, just skip remaining with same length
                            if len(path_shortest) == len(valid_path) - 1:
                                continue

                            # found longer path
                            path_longer = valid_path[1::]
                            result.append(tuple([f[1] for f in path_longer]))
                            return result
                        paths_new.append(valid_path)

        return result

    #
    # RULES
    #

    def reverse_path(self, path):
        return [self.data.inverse_rel(rel) for rel in path[::-1]]

    def generate_rules_for_rel(self, graph_name: Graph_names, rel: int) -> NDArray:
        rules_h1 = self._get_rules_h1(graph_name, rel)
        rules_h2 = self.get_rules_h2(graph_name, rel)
        rules_for_rel = rules_h1.union(rules_h2)

        rules = np.array([self.create_rule(rel, rel_path) for rel_path in rules_for_rel],
                         dtype=self.dtype_rule)

        print(f'- relation \'{rel}\': done')

        return rules

    def get_rules_with_updated_neg_freq(self, rel) -> NDArray:
        rules_data = []
        for rule in self.rules[np.where(self.rules['consequent'] == rel)]:
            neg, freq = self.calculate_neg_freq(Graph_names.Train, rule)
            rules_data.append((rule['consequent'], self.get_rule_antecedent(rule), neg, freq))

        print(f'- neg for relation \'{rel}\': done')

        rules = np.array([self.create_rule(consequent, antecedent, neg=neg, freq=freq)
                          for (consequent, antecedent, neg, freq) in rules_data],
                         dtype=self.dtype_rule)
        return rules

    def get_neg_freq(self, graph, rule, reverse=False):
        a = self.reverse_path(self.get_rule_antecedent(rule)) if reverse else self.get_rule_antecedent(rule)
        rel = self.data.inverse_rel(rule['consequent']) if reverse else rule['consequent']
        all_endpoints = 0
        neg = 0
        for tail, heads in self.data.get_facts_for_rel(graph, rel):
            endpoints = self.find_endpoints_from_node(graph, tail, a)
            neg += sum(found not in heads for found in endpoints)
            all_endpoints += len(endpoints)

        # neg = sum(len(ends) for _, ends in endpoints) - valid
        return neg, all_endpoints

    def calculate_neg_freq(self, graph, rule):
        right_k, freq_r = self.get_neg_freq(graph, rule, True)
        left_k, freq_l = self.get_neg_freq(graph, rule)

        return right_k + left_k, freq_r + freq_l

    def find_endpoints_from_node(self, graph, start, rel_path):
        paths = [[start, ], ]
        updated = []
        for rel in rel_path:
            updated = []
            for visited_entities in paths:
                res = [node for node in self.data.get_heads_for_rel_tail(graph, rel, visited_entities[-1]) if
                       node not in visited_entities]
                updated.extend([visited_entities + [r] for r in res])
            paths = updated
        return [path[-1] for path in updated]

    def get_rule_antecedent(self, rule: NDArray):
        return rule['antecedent'][:rule['length']]

    def print_rules_info(self, rules, verbose=False):
        relations, counts = np.unique(rules['consequent'], return_counts=True)

        print(f'Found {sum(counts)} rules for {len(relations)} relations')
        if verbose:
            for i, r in enumerate(relations):
                print(f'- realtion: \'{r}\': {counts[i]}')

    def create_rule(
            self,
            consequent: int,
            antecedent: tuple[int],
            weight=0.0, freq=0, neg=0, length=0) -> tuple[int, NDArray, float, int, int, int]:

        c = np.empty((self.max_rule_length,), dtype=np.uint16)
        length = min(len(antecedent), self.max_rule_length)
        for i in np.arange(length):
            c[i] = antecedent[i]

        return consequent, c, abs(weight), abs(freq), abs(neg), abs(length)

    def get_rules_for_rel(self, rel: int) -> NDArray:
        return self.rules[self.rules['consequent'] == rel]

    def get_red_k(self, rule, t, edge_dual_var, dual_complexity, graph):
        sum_edges = sum([
            self.path_exists(self.get_rule_antecedent(rule), edge, graph) * dual_var for edge, dual_var in
            edge_dual_var.items()
        ])
        return t * rule['neg'] - sum_edges - (1 + rule['length']) * dual_complexity

    def get_dual_vars(self, constraints):
        dual_complex = constraints['c_complexity'].pi
        dual_edges = {}
        constraints.pop('c_complexity')

        for name, c in list(constraints.items()):
            tail_head = re.findall('\d+', name)
            assert len(tail_head) == 2
            dual_edges[tuple([int(x) for x in tail_head])] = c.pi

        return dual_complex, dual_edges

    #
    # LP METHODS
    #
    def get_solver(self):
        # find available solver
        self.solver_name = DEFAULT_SOLVER
        if self.args.solver in pl.listSolvers(onlyAvailable=True):
            self.solver_name = self.args.solver
            print(f'Using solver \'{self.solver_name}\'')
        else:
            print(f'Solver \'{self.args.solver}\' not available, using default: \'{self.solver_name}\'')

        # works for default solver, some config needed for others
        solver = pl.getSolver(self.solver_name, msg=0)
        return solver

    def solve_for_rel(self, rel: int, solver: pl.LpSolver) -> NDArray:
        rules_for_rel = self.rules[self.rules['consequent'] == rel]
        k = np.max(rules_for_rel['length'])
        max_complexity = rules_for_rel.size + np.sum(rules_for_rel['length'])

        mrr_weights = self.solve_lps(solver, rel, k, max_complexity, Graph_names.Train, Graph_names.Validate)
        print(f'rel {rel}: all LPs solved')
        print(f'- MRR: {mrr_weights[0]}')

        print('================================================\n\n================================================')
        rules = []
        for (weight, rule) in mrr_weights[1]:
            r = rule
            r['weight'] = weight
            rules.append(r)
        rules = np.array(rules, dtype=self.dtype_rule)

        return rules

    def solve_lps(self, solver: pl.LpSolver, rel: int, k: int, max_complexity: int, graph_lp: Graph_names,
                  graph_mrr: Graph_names):
        base_problem, weights_rules, penalty = self._init_base_problem(rel, graph_lp)
        result = []
        # MRR -1 so first combination of weihts is returned even if all return MRR 0 
        mrr_weights = (-1, [])
        found_combinations = set()
        for t in self.tradeoff:
            for i in np.arange(1, self.iterations + 1):
                complexity = i * k
                problem, updated_weights = self.create_lp_problem(base_problem, weights_rules.copy(), penalty.copy(),
                                                                  rel, t, complexity)
                problem.solve(solver)

                # save weight > 0
                weights_to_update = []
                weights = []
                for w, rule in updated_weights:
                    if w.value() > 0:
                        weights_to_update.append((w.value(), rule))
                        weights.append(w)

                weights = tuple(weights)
                # no reason to save duplicate combinations of rules
                if weights in found_combinations:
                    continue

                found_combinations.add(weights)
                result.append(((t, i), weights_to_update))

            print(f'rel: {rel}, t: {t} done')

        print(f'Weights for {rel} calculated')
        # sort by i (desc), lowest number of rules selected for same MRR
        result.sort(key=lambda res: res[0][1])
        combination_count = len(result)
        for i, ti_weights in enumerate(result):
            mrr = self.mrr_for_rules(rel, ti_weights[1], graph_mrr)
            print(f'{i + 1} / {combination_count} : {mrr}')
            if mrr > mrr_weights[0]:
                mrr_weights = (mrr, ti_weights[1])

        return mrr_weights

    def init_custom_problem(self, rel: int, rules, graph: Graph_names, t, complexity):
        problem = pl.LpProblem(f'Problem_{rel}', pl.LpMinimize)
        constraints = {}
        penalties = []
        weights_rules = [(pl.LpVariable(f'w_{i}', lowBound=0.0, upBound=1.0), rule) for i, rule in enumerate(rules)]

        for tail, ends in self.data.get_facts_for_rel(graph, rel):
            for head in ends:
                penalty = pl.LpVariable(f'Penalty_{tail}:{head}', lowBound=0.0, upBound=1.0)
                penalties.append(penalty)
                constraint = (
                                     pl.lpSum(
                                         [int(self.path_exists(self.get_rule_antecedent(rule), (tail, head),
                                                               graph)) * weight
                                          for weight, rule in weights_rules]
                                     ) + penalty
                             ) >= 1.0
                constraints[(tail, head)] = constraint
                problem += (constraint, f'c_{tail}:{head}')

        c_complex = (pl.lpSum(
            [(1 + rule['length']) * weight for weight, rule in weights_rules])
                    ) <= complexity
        problem += (c_complex, 'c_complexity')

        obj_func = pl.lpSum(penalties) + t * pl.lpSum([rule['neg'] * weight for weight, rule in weights_rules])
        problem += (obj_func, 'obj_func')

        return problem, weights_rules, constraints

    def _init_base_problem(self, rel: int, graph: Graph_names):
        base_problem = pl.LpProblem(f'Problem_{rel}', pl.LpMinimize)
        penalties = []
        weights_rules = [(pl.LpVariable(f'w_{i}', lowBound=0.0, upBound=1.0), rule) for i, rule in
                         enumerate(self.get_rules_for_rel(rel))]

        for tail, ends in self.data.get_facts_for_rel(graph, rel):
            for head in ends:
                penalty = pl.LpVariable(f'Penalty_{tail}:{head}', lowBound=0.0, upBound=1.0)
                penalties.append(penalty)
                constraint = (
                                     pl.lpSum(
                                         [int(self.path_exists(self.get_rule_antecedent(rule), (tail, head),
                                                               graph)) * weight
                                          for weight, rule in weights_rules]
                                     ) + penalty
                             ) >= 1.0
                base_problem += (constraint, f'c_{tail}:{head}')

        return base_problem, weights_rules, penalties

    def create_lp_problem(self, base_problem: pl.LpProblem, weights, penalty, rel: int, t: int, complexity: int):
        # prediction function:
        # f_r(X,Y) = sum_i=1..p(w_i * C_i(X,Y) for each X,Y)

        # (1) z_min = sum_i=1..m(eta_i) + tau * sum_k∈K(neg_k * w_k)
        # m ...... number of edges labeled by relation r
        # eta_i .. penalty
        #          - positive if prediction function defined by rules with w_k > 0 for i-th edge in E_r gives value < 1
        # tau .... tradeoff between how well our weighted combination of rules performs on the known
        #          facts (gives positive scores), and how poorly it performs on some negative samples or “unknown"
        #          facts - list of recommended values per dataset
        # K ...... clauses (rules/antecedents)

        # (2) sum_k∈K(a_ik * w_k) + eta_i >= 1 for each i ∈ E_r
        # a_ik .. 1 if relational path from k-th rule exists for i-th edge in E_r

        # (3) sum_k∈K((1 + |C_k|) * w_k) <= kappa
        # |C_k| .. number of rules for rel r? length of rule antecedent.. hopefully
        # kappa .. upper bound of complexity (number of clauses + number of relations across all clauses)

        # w_k in [0,1]

        problem = base_problem.copy()

        c_complex = (pl.lpSum(
            [(1 + rule['length']) * weight for weight, rule in weights])
                    ) <= complexity
        problem += (c_complex, 'c_complexity')

        obj_func = pl.lpSum(penalty) + t * pl.lpSum([rule['neg'] * weight for weight, rule in weights])
        problem += (obj_func, 'obj_func')

        return problem, weights

    #
    # RANKING
    #
    def triple_rank_precomputed(self, fact: tuple[int, int, int], graph, scores_right, weights_paths, rel_paths_rev):
        score_l = self.entity_score(graph, rel_paths_rev, fact[::-2])
        score_r = self.entity_score(graph, weights_paths, fact[::2])

        # left
        entities_l = self.data.get_heads_corrupted_for_rel_tail(graph, self.data.inverse_rel(fact[1]), fact[2])
        scores_left = [self.entity_score(graph, rel_paths_rev, (fact[2], entity)) for entity in entities_l]

        rank_l = self.get_rank(score_l, scores_left)
        rank_r = self.get_rank(score_r, scores_right)

        return int((rank_l + rank_r) / 2)

    def get_rank_pesimist(self, entity_score, other_scores):
        rank = 1
        for score in other_scores:
            rank += score > entity_score

        return rank

    def get_rank(self, entity_score, other_scores):
        rank = 1
        count_equal = 0
        for score in other_scores:
            rank += score > entity_score
            count_equal += entity_score == score

        # rank += sum(score > entity_score for score in other_scores)
        rank += sum(self.rng.integers(2, size=count_equal))
        return rank

    def entity_score(self, graph, weights_paths, fact: tuple[int, int]):
        return sum(weight * self.path_exists(path, fact, graph) for weight, path in weights_paths)

    def calculate_MRR(self, ranks: list[int]) -> float:
        mrr = 0
        if len(ranks) == 0:
            return mrr

        mrr = sum(1 / rank for rank in ranks)
        mrr /= len(ranks)

        return mrr

    def mrr_for_rules(self, rel: int, weights_rules, graph: Graph_names):
        rules = []
        for weight, rule in weights_rules:
            r = rule
            r['weight'] = weight
            rules.append(r)

        weights_paths = [(r['weight'], self.get_rule_antecedent(r)) for r in rules]
        rel_paths_rev = []
        for weight, path in weights_paths:
            rel_paths_rev.append((weight, [self.data.inverse_rel(rel) for rel in path[::-1]]))
        ranks = []

        # (t,r,?)
        for tail, heads in self.data.get_facts_for_rel(graph, rel):
            entities_r = self.data.get_heads_corrupted_for_rel_tail(graph, rel, tail)
            scores = np.zeros(self.data.num_entity)
            for rule in rules:
                endpoints = self.find_endpoints_from_node(graph, tail, self.get_rule_antecedent(rule))
                scores[endpoints] += rule['weight']
            scores_filtered = scores[entities_r]
            for h in heads:
                score_h = scores[h]
                rank = self.get_rank(score_h, scores_filtered)
                ranks.append(rank)

        # (?, r, h)
        for head, tails in self.data.get_facts_for_rel(graph, self.data.rel_to_inv(rel)):
            entities_l = self.data.get_heads_corrupted_for_rel_tail(graph, self.data.rel_to_inv(rel), head)
            scores = np.zeros(self.data.num_entity)
            for weight, rev_path in rel_paths_rev:
                endpoints = self.find_endpoints_from_node(graph, head, rev_path)
                scores[endpoints] += weight
            scores_filtered = scores[entities_l]
            for t in tails:
                score_t = scores[t]
                rank = self.get_rank(score_t, scores_filtered)
                ranks.append(rank)
        mrr = self.calculate_MRR(ranks)

        return mrr

    def get_mrr_and_ranks(self, rel: int, graph: Graph_names) -> tuple[int, float, list[int]]:
        """Returns relation, mrr and list of ranks for single relation"""
        ranks = []
        rules = self.get_rules_for_rel(rel)
        weights_paths = [(r['weight'], self.get_rule_antecedent(r)) for r in rules]
        rel_paths_rev = []
        for weight, path in weights_paths:
            rel_paths_rev.append((weight, [self.data.inverse_rel(rel) for rel in path[::-1]]))

        # (t,r,?)
        for tail, heads in self.data.get_facts_for_rel(graph, rel):
            entities_r = self.data.get_heads_corrupted_for_rel_tail(graph, rel, tail)
            scores = np.zeros(self.data.num_entity)
            for rule in rules:
                endpoints = self.find_endpoints_from_node(graph, tail, self.get_rule_antecedent(rule))
                scores[endpoints] += rule['weight']
            scores_filtered = scores[entities_r]
            for h in heads:
                score_h = scores[h]
                rank = self.get_rank(score_h, scores_filtered)
                ranks.append(rank)

        # (?, r, h)
        for head, tails in self.data.get_facts_for_rel(graph, self.data.rel_to_inv(rel)):
            entities_l = self.data.get_heads_corrupted_for_rel_tail(graph, self.data.rel_to_inv(rel), head)
            scores = np.zeros(self.data.num_entity)
            for weight, rev_path in rel_paths_rev:
                endpoints = self.find_endpoints_from_node(graph, head, rev_path)
                scores[endpoints] += weight
            scores_filtered = scores[entities_l]
            for t in tails:
                score_t = scores[t]
                rank = self.get_rank(score_t, scores_filtered)
                ranks.append(rank)

        mrr = self.calculate_MRR(ranks)
        print(f'rel: {rel}: done')

        return rel, mrr, ranks

    def hits_k(self, ranks, k) -> float:
        hits = 0
        if len(ranks) == 0:
            return hits

        # for rank in ranks:
        #     if rank <= k:
        #         hits += 1

        hits = sum(rank <= k for rank in ranks)
        return hits / len(ranks)

    #
    # OTHER
    #
    def can_append_to_path(self, path: list[tuple[int, int, int]], fact: tuple[int, int, int]) -> bool:
        """Determines if fact can be appended to path where first element is starting fact not included in final path"""
        if len(path) == 1:
            # first step - same tail as original
            return (path[0] != fact) & (path[0][0] == fact[0])

        # check if entity visited
        for entities in [step[::2] for step in path[1::]]:
            if fact[2] in entities:
                return False

        # return fact[0] != path[0][2]
        return True

    def path_exists(self, rel_path: NDArray, fact: tuple[int, int], graph: Graph_names):
        return fact[1] in self.find_endpoints_from_node(graph, fact[0], rel_path)
