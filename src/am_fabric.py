#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.am_graph import AMGraph, EdgeType, Por, NodeId, NodeKey, get_node_id
from src.normalise_amgraph import NormaliseAMGraph
from src.am_gas import AMGas
from typing import Optional, Set, Tuple, List
from copy import deepcopy


class AMFabric(object):
    def __init__(self,
                 fabric_name: str,
                 anomaly_threshold_factor: float = 4.0,
                 fast_alpha: float = 0.7,
                 prune_threshold: float = 0.01,
                 audit: bool = False,
                 normalise: bool = True):

        self.fabric_name: str = fabric_name
        self.anomaly_threshold_factor: float = anomaly_threshold_factor
        self.fast_alpha: float = fast_alpha
        self.slow_alpha: float = 0.1
        self.prune_threshold: float = prune_threshold
        self.audit: bool = audit
        self.normalise: bool = normalise
        self.domains = {}

    def calc_communities(self):
        for domain in self.domains:
            self.domains[domain].calc_communities()

    def to_dict(self, denormalise: bool = False) -> dict:

        fabric_dict = {}
        for domain in self.domains:

            fabric_dict[domain] = self.domains[domain].to_dict(denormalise=denormalise)

        return fabric_dict

    def train_domain(self,
                     training_graph: AMGraph,
                     ref_id: str,
                     search_edge_types: Set[EdgeType],
                     learn_edge_types: Set[EdgeType],
                     domain: str = 'temporal') -> Por:

        if domain not in self.domains:
            self.domains[domain] = {'gas': AMGas(fabric_name=self.fabric_name,
                                                 domain=domain,
                                                 anomaly_threshold_factor=self.anomaly_threshold_factor,
                                                 fast_alpha=self.fast_alpha,
                                                 prune_threshold=self.prune_threshold,
                                                 audit=self.audit,
                                                 normalise=self.normalise),
                                    'stm': AMGraph(),
                                    'ltm': AMGraph()
                                    }

        # prepare the temporal graph tp associate with the incoming data
        #
        ltm = AMGraph(graph=self.domains[domain]['ltm'])
        ltm.rename_triples(postfix_edge_uid='ltm')
        stm = AMGraph(graph=self.domains[domain]['stm'])
        stm.rename_triples(postfix_edge_uid='stm')

        # combine the short-term, long-term and current data into 1 graph
        #
        t_graph = AMGraph(graph=training_graph)
        t_graph.merge_graph(graph_to_merge=ltm, weight=1.0)
        t_graph.merge_graph(graph_to_merge=stm, weight=1.0)

        search_edge_types.update(ltm.edgeTypes)
        search_edge_types.update(stm.edgeTypes)
        learn_edge_types.update(ltm.edgeTypes)
        learn_edge_types.update(stm.edgeTypes)

        por = self.domains[domain].train(training_graph=t_graph,
                                         ref_id=ref_id,
                                         search_edge_types=search_edge_types,
                                         learn_edge_types=learn_edge_types)

        self.domains[domain]['stm'].learn_graph(graph_to_learn=training_graph,
                                                learn_rate=self.fast_alpha,
                                                learn_edge_types=search_edge_types)

        self.domains[domain]['ltm'].learn_graph(graph_to_learn=training_graph,
                                                learn_rate=self.slow_alpha,
                                                learn_edge_types=search_edge_types)

        return por

    def train(self,
              training_graph: AMGraph,
              ref_id: str,
              search_edge_types: Set[EdgeType],
              learn_edge_types: Set[EdgeType]) -> Por:

        por = {}
        association_graph = AMGraph()
        for sub_graph in training_graph.get_sub_graphs(generalise=True):
            por[sub_graph[0]] = self.train_domain(training_graph=sub_graph[1],
                                                  ref_id=ref_id,
                                                  search_edge_types=search_edge_types,
                                                  learn_edge_types=learn_edge_types,
                                                  domain=sub_graph[0])

            # build up the association graph from either the BMU or new neuron
            #
            if por[sub_graph[0]]['new_neuron_key'] is not None:
                association_graph.set_edge(triple=(('Association', '1'), ('ASSOCIATED_WITH', None, None), (sub_graph[0], por[sub_graph[0]]['new_neuron_key'])))
            else:
                association_graph.set_edge(triple=(('Association', '1'), ('ASSOCIATED_WITH', None, None), (sub_graph[0], por[sub_graph[0]]['bmu_key'])))

        por['ASSOCIATION'] = self.train_domain(training_graph=association_graph,
                                               ref_id=ref_id,
                                               search_edge_types={'ASSOCIATED_WITH'},
                                               learn_edge_types={'ASSOCIATED_WITH'},
                                               domain='ASSOCIATION')

        return por

    def query_domain(self, query_graphs: List[AMGraph], domain: str = 'temporal', bmu_only: bool = True) -> Por:

        ltm = AMGraph()
        stm = AMGraph()

        for query_graph in query_graphs:

            stm.learn_graph(graph_to_learn=query_graph,
                            learn_rate=self.fast_alpha,
                            learn_edge_types=query_graph.edgeTypes)

            ltm.learn_graph(graph_to_learn=query_graph,
                            learn_rate=self.slow_alpha,
                            learn_edge_types=query_graph.edgeTypes)

        ltm.rename_triples(postfix_edge_uid='ltm')
        stm.rename_triples(postfix_edge_uid='stm')
        q_graph = AMGraph(graph=ltm)
        q_graph.merge_graph(graph_to_merge=stm, weight=1.0)

        por = self.domains[domain].query(query_graph=q_graph, bmu_only=bmu_only)

        return por


