from os import path
from enum import Enum

import numpy as np
from numpy._typing import NDArray
from scipy.sparse import dok_array

filename_test = "test.txt"
filename_train = "train.txt"
filename_valid = "valid.txt"
filename_entities = "entities.txt"
filename_relations = "relations.txt"

def generate_files(dataset_dir) -> None:
    entities = set()
    relations = set()
    for filepath in [filename_test, filename_train, filename_valid]:
        full_path = path.join(dataset_dir, filepath)
        if not path.isfile(full_path):
            print(f'File \'{full_path}\' not found while creating entities and relations files')
            continue
        with open(full_path) as f:
            for line in f:
                splitted = line.strip().split()
                assert(len(splitted) == 3)

                # expected triplets in form (head, relation, tail)
                relations.add(splitted[1])
                entities.add(splitted[0])
                entities.add(splitted[2])

    # write to files
    with open(path.join(dataset_dir, filename_relations), 'w') as f:
        f.writelines([relation + '\n' for relation in relations])

    with open(path.join(dataset_dir, filename_entities), 'w') as f:
        f.writelines([entity + '\n' for entity in entities])

def get_element_id(element: str):
    return int(''.join(filter(str.isdigit, element)))

def num_encode(file_path: str) -> dict[str, int]:
    n = 0
    result = {}
    with open(file_path) as f:
        for line in f:
            splitted = line.strip().split()
            assert len(splitted) == 1

            result[splitted[0]] = n
            n += 1
    return result



class Graph_names(Enum):
    Train = 'train',
    Test = 'test',
    Validate = 'validate'

class Data:
    def __init__(self, dataset_dir) -> None:
        self.file_test = path.join(dataset_dir, filename_test)
        self.file_train = path.join(dataset_dir, filename_train)
        self.file_validate = path.join(dataset_dir, filename_valid)

        self.file_relations = path.join(dataset_dir, filename_relations)
        self.file_entities = path.join(dataset_dir, filename_entities)

        self.entity_to_num = num_encode(self.file_entities)
        self.relation_to_num = num_encode(self.file_relations)
        self.num_to_entity = {num: entity for entity, num in self.entity_to_num.items()}
        self.num_to_relation = {num: relation for relation, num in self.relation_to_num.items()}

        self.num_entity = len(self.entity_to_num)
        self.num_relation = len(self.relation_to_num)

        self.graphs = {
            Graph_names.Train    : self.create_graph(self.file_train, include_inverse=True),
            Graph_names.Validate : self.create_graph(self.file_validate, True),
            Graph_names.Test     : self.create_graph(self.file_test, True)
        }

    #
    # GRAPH
    #
    def create_graph(self, filepath, include_inverse=False):
        graph = {}
        for rel in np.arange(self.num_relation * (include_inverse + 1)):
            graph[rel] = {}
            for e in np.arange(self.num_entity):
                graph[rel][e] = []

        with open(filepath) as f:
            for line in f:
                splitted = line.strip().split()
                assert len(splitted) == 3

                tail = self.entity_to_num[splitted[0]]
                rel = self.relation_to_num[splitted[1]]
                head = self.entity_to_num[splitted[2]]

                try:
                    graph[rel][head].append(tail)
                except KeyError:
                    graph[rel][head] = [tail, ]

                if include_inverse:
                    try:
                        graph[rel + self.num_relation][tail].append(head)
                    except:
                        graph[rel + self.num_relation][tail] = [head,]

        return graph
    
    def create_graph_sparse(self, filepath, include_inverse=False):
        graph = {}
        for rel in np.arange(self.num_relation):
            graph[rel] = dok_array((self.num_entity, self.num_entity), dtype=np.bool_)

        with open(filepath) as f:
            for line in f:
                splitted = line.strip().split()
                assert len(splitted) == 3

                tail = self.entity_to_num[splitted[0]]
                rel = self.relation_to_num[splitted[1]]
                head = self.entity_to_num[splitted[2]]

                graph[rel][tail, head] = True


        if include_inverse:
            for rel in np.arange(self.num_relation):
                graph[rel+self.num_relation] = graph[rel].transpose()

        for rel in graph:
            graph[rel] = graph[rel].tolil()

        return graph

    def get_graph(self, name: Graph_names):
        return self.graphs[name]

    #
    # INVERSE
    #
    def is_inverse_included(self, name: Graph_names):
        return len(self.graphs[name]) > self.num_relation

    def rel_to_inv(self, rel) -> int:
        if rel < 0:
            self.rel_to_inv(abs(rel))
        if not self.is_inverse(rel):
            return rel + self.num_relation
        return rel

    def inv_to_rel(self, inv) -> int:
        if inv < 0:
            self.inv_to_rel(abs(inv))
        if self.is_inverse(inv):
            return inv % self.num_relation
        return inv
    
    def inverse_rel(self, rel):
        return self.inv_to_rel(rel) if self.is_inverse(rel) else self.rel_to_inv(rel)

    def is_inverse(self, rel) -> bool:
        return abs(rel) >= self.num_relation

    #
    # TAILS
    #

    def get_tails_for_rel_head(self, graph, rel, head):
        return self.graphs[graph][rel][head]
    
    def find_heads_for_rel_tail(self, graph: Graph_names, rel: int, tail: int):
        return [head for head, tails in self.graphs[graph][rel].items() if tail in tails]

    def get_tails_for_rel_entity(self, graph: Graph_names, rel: int, entity: int, reverse=False) -> list[int]:
        result = None

        if self.is_inverse_included(graph):
            if reverse:
                r = self.inv_to_rel(rel) if self.is_inverse(rel) else self.rel_to_inv(rel)
                result = self.get_tails_for_rel_head(graph, r, entity)
            else:
                result = self.get_tails_for_rel_head(graph, rel, entity)
        else:
            transpose_data = self.is_inverse(rel)
            r = self.inv_to_rel(rel)
            if reverse:
                transpose_data = not transpose_data
            if transpose_data:
                result = self.find_heads_for_rel_tail(graph, r, entity)
            else:
                result = self.get_tails_for_rel_head(graph, r, entity)

        return result

    def get_tails_corrupted_for_rel_head(self, graph_name: Graph_names, rel: int, head: int, transpose=False):
        indexes = self.get_tails_for_rel_head(graph_name, rel, head)
        mask = np.ones(self.num_entity, dtype=bool)
        mask[indexes] = False
        return np.array(np.arange(self.num_entity))[mask]

    def get_tails_corrupted_for_rel_entity(self, graph: Graph_names, rel: int, entity: int, reverse=False) -> NDArray[np.intp]:
        """Same as 'get_tails_for_rel_entity' but only not existing facts are returned"""
        result = None

        if self.is_inverse_included(graph):
            if reverse:
                r = self.inverse_rel(rel)
                result = self.get_tails_corrupted_for_rel_head(graph, r, entity)
            else:
                result = self.get_tails_corrupted_for_rel_head(graph, rel, entity)
        else:
            transpose_data = self.is_inverse(rel)
            r = self.inv_to_rel(rel)
            if reverse:
                transpose_data = not transpose_data
            if transpose_data:
                result = self.get_tails_corrupted_for_rel_head(graph, r, entity, True)
            else:
                result = self.get_tails_corrupted_for_rel_head(graph, r, entity)

        return result

    def get_tails(self, graph, head):
        return [(rel, self.graphs[graph][rel][head]) for rel in self.graphs[graph] if self.graphs[graph][rel][head]]

    #
    # FACTS
    #
    def get_facts_for_rel(self, graph, rel):
        return [(head, tails) for (head, tails) in self.graphs[graph][rel].items() if tails]
