#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from sogam.am_graph import AMGraph, EdgeType, Por, NodeId, NodeKey, get_node_id
from sogam.normalise_amgraph import NormaliseAMGraph
from sogam.am_gas import AMGas
from typing import Optional, Set, Tuple, List
from copy import deepcopy





class AMFabric(object):
    def __init__(self,
                 fabric_name: str,
                 anomaly_threshold_factor: float = 4.0,
                 fast_alpha: float = 0.9,
                 prune_threshold: float = 0.01,
                 audit: bool = False,
                 normalise: bool = True,
                 ):

        self.fabric_name: str = fabric_name
        self.anomaly_threshold_factor: float = anomaly_threshold_factor
        self.fast_alpha: float = fast_alpha
        self.slow_alpha: float = 0.1
        self.prune_threshold: float = prune_threshold
        self.audit: bool = audit
        self.normalise: bool = normalise

        self.domains = {'temporal': AMGas(fabric_name=self.fabric_name,
                                          domain='temporal',
                                          anomaly_threshold_factor=self.anomaly_threshold_factor,
                                          fast_alpha=fast_alpha,
                                          prune_threshold=self.prune_threshold,
                                          audit=audit,
                                          normalise=normalise)
                        }

        self.ltm = AMGraph()
        self.stm = AMGraph()

    def calc_communities(self):
        self.domains['temporal'].calc_communities()

    def to_dict(self, denormalise: bool = False) -> dict:
        return self.domains['temporal'].to_dict(denormalise=denormalise)

    def train(self,
              training_graph: AMGraph,
              ref_id: str,
              search_edge_types: Set[EdgeType],
              learn_edge_types: Set[EdgeType]) -> Por:

        ltm = AMGraph(graph=self.ltm)
        ltm.rename_triples(postfix_edge_uid='ltm')
        stm = AMGraph(graph=self.stm)
        stm.rename_triples(postfix_edge_uid='stm')
        t_graph = AMGraph(graph=training_graph)
        t_graph.merge_graph(graph_to_merge=ltm, weight=1.0)
        t_graph.merge_graph(graph_to_merge=stm, weight=1.0)

        search_edge_types.update(ltm.edgeTypes)
        search_edge_types.update(stm.edgeTypes)
        learn_edge_types.update(ltm.edgeTypes)
        learn_edge_types.update(stm.edgeTypes)

        por = self.domains['temporal'].train(training_graph=t_graph,
                                             ref_id=ref_id,
                                             search_edge_types=search_edge_types,
                                             learn_edge_types=learn_edge_types)

        self.stm.learn_graph(graph_to_learn=training_graph,
                             learn_rate=self.fast_alpha,
                             learn_edge_types=search_edge_types)

        self.ltm.learn_graph(graph_to_learn=training_graph,
                             learn_rate=self.slow_alpha,
                             learn_edge_types=search_edge_types)

        return por

    def query(self, query_graphs: List[AMGraph], bmu_only: bool = True) -> Por:

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

        por = self.domains['temporal'].query(query_graph=q_graph, bmu_only=bmu_only)

        return por


