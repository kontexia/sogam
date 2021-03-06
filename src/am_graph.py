#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict, Tuple, Union, Set
from copy import deepcopy
from math import pow, sqrt
import time
from threading import Lock
import networkx as nx


# alias types
#
NodeType = str
""" the type of a node """

NodeUid = str
""" a node' unique identifier within the context of the node's type """

NodeId = Tuple[NodeType, NodeUid]
""" the unique identifier of a node - the concatenation of its' type and uid"""

NodeKey = str
""" jason compliant unique identifier of a node used in hashes """

SourceId = NodeId
""" unique identifier of the source node in a triple """

TargetId = NodeId
""" unique identifier of the target node in a triple """

EdgeType = str
""" the type of an edge """

EdgeUid = str
""" an edge's unique identifier within the context of the edge's type """

EdgeExpired = float
""" the timestamp when an edge has expired """

EdgeId = Tuple[EdgeType, Optional[EdgeUid], Optional[EdgeExpired]]
""" the unique identifier of an edge withing the context of the edge type - a concatenation of Edge type, uid, expired timestamp"""

TripleId = Tuple[SourceId, EdgeId, TargetId]
""" the unique identifier of a triple source node, edge and target node """

TripleKey = str
""" json compliant unique identifier of a triple used in hashes """

Por = dict
""" dictionary representing the path of reasoning """


def get_node_key(node_id: NodeId) -> NodeKey:
    # ':' special - enables easy splitting of keys to create ids
    #
    return f'{node_id[0]}:{node_id[1]}'


def get_node_id(node_key: NodeKey) -> NodeId:
    # assumes ':' delimits type and uid
    #
    node = node_key.split(':')
    return node[0], node[1]


def get_triple_key(triple_id: TripleId, directional: bool = False) -> TripleKey:
    source_key = get_node_key(node_id=triple_id[0])
    target_key = get_node_key(node_id=triple_id[2])

    # if not directional then key will consist of alphanumeric sort of source_key and target_key
    # note ':' special - delimits all components and allows for easy splitting to derive equivalent id
    #
    if (not directional and source_key < target_key) or directional:
        triple_key = f'{source_key}:{triple_id[1][0]}:{triple_id[1][1]}:{triple_id[1][2]}:{target_key}'
    else:
        triple_key = f'{target_key}:{triple_id[1][0]}:{triple_id[1][1]}:{triple_id[1][2]}:{source_key}'
    return triple_key


def get_triple_id(triple_key: TripleKey) -> TripleId:
    # assume special ':' character delimits components
    #
    triple = triple_key.split(':')

    # Note have to deal with converting string 'None' to type None
    #
    return (triple[0], triple[1]), (triple[2], triple[3] if triple[3] != 'None' else None, float(triple[4]) if triple[4] != 'None' else None), (triple[5], triple[6])


class AMGraph(object):
    """
    class to represent a sparse graph of nodes connected by edges
    """

    # class attribute representing next unique identifier for an instance of AMGraph
    # with associated thread lock
    #
    _next_uid = None
    _uid_lock = Lock()

    def __init__(self, uid: Optional[str] = None, graph=None, directional: bool = True, uid_copy: bool = True, normalised: bool = False):
        """
        AMGraph represents an sparse graph of nodes connected via edges and capable of performing graph comparisons, learning and merging
        :param uid: optional unique identifier of a graph, if None then autogenerated
        :param graph: optional AMGraph or Dict that will be copied
        :param directional: Boolean flag indicating if edges are directional
        :param uid_copy: Boolean flag indicating if uid of supplied graph should be copied
        :param normalised: Boolean flag indicating edges of graph are normalised
        """

        # the edges keyed by EdgeKey
        #
        self.edges: dict = {}

        # the nodes keyed by NodeKey
        #
        self.nodes: dict = {}

        self.directional = directional
        self.normalised = normalised

        # dict of NodeKeys with attributes that are graphs
        #
        self.embedded_graphs = {}

        self.edgeTypes = set()

        self.uid = None

        if graph is not None:
            if isinstance(graph, AMGraph):
                self.edges = deepcopy(graph.edges)
                self.nodes = deepcopy(graph.nodes)
                self.embedded_graphs = deepcopy(graph.embedded_graphs)
                self.normalised = graph.normalised
                self.directional = graph.directional
                self.edgeTypes = deepcopy(graph.edgeTypes)
                if uid_copy:
                    self.uid = graph.uid

            elif isinstance(graph, dict):
                self.edges = deepcopy(graph['edges'])
                self.nodes = deepcopy(graph['nodes'])
                self.directional = graph['directional']
                self.embedded_graphs = deepcopy(graph['embedded_graphs'])
                self.normalised = graph['normalised']
                self.edgeTypes = deepcopy(graph['edgeTypes'])

                if uid_copy:
                    self.uid = graph['uid']

                # reconstruct any embedded graphs
                #
                for node_key in self.embedded_graphs:
                    for attr in self.embedded_graphs[node_key]:
                        self.nodes[node_key][attr] = AMGraph(graph=self.nodes[node_key][attr])

        # set the uid if provided
        #
        if uid is not None:
            self.uid = uid

        # else if graph has not been provided to copy or copy_uid is not required
        #
        if self.uid is None:

            # if the class next_uid is None then start from 1
            #
            with AMGraph._uid_lock:

                # set to 1 if never been set before
                #
                if AMGraph._next_uid is None:
                    AMGraph._next_uid = 1

                # create standard uid
                #
                self.uid = f'_graph_{AMGraph._next_uid}'

                # increment for next graphs
                #
                AMGraph._next_uid += 1

    def to_dict(self, denormaliser=None) -> dict:
        """
        represents graphs as a dictionary
        :return: dictionary with keys: edges, nodes, uid, directional
        """
        graph_dict = {'nodes': deepcopy(self.nodes),
                      'edges': deepcopy(self.edges),
                      'edgeTypes': deepcopy(self.edgeTypes),
                      'directional': self.directional,
                      'normalised': self.normalised,
                      'uid': self.uid,
                      'embedded_graphs': deepcopy(self.embedded_graphs),
                      'amgraph': True}

        for node_key in self.embedded_graphs:
            for attr in self.embedded_graphs[node_key]:
                if denormaliser is not None:
                    graph_dict['nodes'][node_key][attr] = denormaliser.denormalise(graph=graph_dict['nodes'][node_key][attr]).to_dict(denormaliser=denormaliser)
                    graph_dict['normalised'] = False
                else:
                    graph_dict['nodes'][node_key][attr] = graph_dict['nodes'][node_key][attr].to_dict()

        return graph_dict

    def set_node(self,
                 node: Union[NodeKey, NodeId],
                 timestamp: Optional[float] = None,
                 **node_attributes) -> NodeKey:

        if isinstance(node, tuple):
            node_key = get_node_key(node_id=node)
            node_id = node
        else:
            node_key = node
            node_id = get_node_id(node_key=node)

        if timestamp is None:
            ts = time.time()
        else:
            ts = timestamp

        if node_key not in self.nodes:
            self.nodes[node_key] = {'_type': node_id[0],
                                    '_uid': node_id[1],
                                    '_created': ts,
                                    '_updated': None,
                                    '_edges': set(),
                                    '_community': None,
                                    '_changed': False}

        else:
            self.nodes[node_key]['_updated'] = ts
            self.nodes[node_key]['_changed'] = True

        if len(node_attributes) > 0:
            self.nodes[node_key].update(**node_attributes)

            # keep track of any attributes that are embedded graphs
            #
            for attr in node_attributes:
                if isinstance(node_attributes[attr], AMGraph):
                    if node_key not in self.embedded_graphs:
                        self.embedded_graphs[node_key] = {attr}
                    else:
                        self.embedded_graphs[node_key].add(attr)

        return node_key

    def set_edge(self,
                 triple: Union[TripleKey, TripleId],
                 source_attr: Optional[dict] = None,
                 target_attr: Optional[dict] = None,
                 prob: float = 1.0,
                 numeric: Optional[float] = None,
                 audit: bool = False,
                 timestamp: Optional[float] = None,
                 **edge_attributes
                 ):

        if timestamp is None:
            ts = time.time()
        else:
            ts = timestamp

        if isinstance(triple, tuple):
            triple_id = triple
            triple_key = get_triple_key(triple_id=triple, directional=self.directional)
        else:
            triple_id = get_triple_id(triple_key=triple)
            triple_key = triple

        # keep track of edge types
        #
        self.edgeTypes.add(triple_id[1][0])

        # add nodes if necessary
        #
        if source_attr is not None:
            source_key = self.set_node(node=triple_id[0], timestamp=ts, **source_attr)
        else:
            source_key = self.set_node(node=triple_id[0], timestamp=ts)

        if target_attr is not None:
            target_key = self.set_node(node=triple_id[2], timestamp=ts, **target_attr)
        else:
            target_key = self.set_node(node=triple_id[2], timestamp=ts)

        if audit and triple_key in self.edges:

            # construct an expired triple_id
            #
            expired_triple_id = (triple_id[0], (triple_id[1][0], triple_id[1][1], ts), triple_id[2])
            expired_edge_key = get_triple_key(triple_id=expired_triple_id, directional=self.directional)

            # copy over attributes
            #
            self.edges[expired_edge_key] = deepcopy(self.edges[triple_key])

            # update the attributes
            #
            self.edges[expired_edge_key]['_changed'] = True
            self.edges[expired_edge_key]['_updated'] = ts
            self.edges[expired_edge_key]['_expired'] = ts
            add_new_edge = True

        elif triple_key in self.edges:

            add_new_edge = False

            # update the attributes
            #
            self.edges[triple_key]['_updated'] = ts
            self.edges[triple_key]['_changed'] = True
            self.edges[triple_key]['_prob'] = prob

            if numeric is not None:
                self.edges[triple_key]['_numeric'] = numeric

            if len(edge_attributes) > 0:
                self.edges[triple_key].updated(**edge_attributes)
        else:
            add_new_edge = True

        # add new edge if required
        #
        if add_new_edge:
            self.edges[triple_key] = {'_type': triple_id[1][0],
                                      '_uid': triple_id[1][1],
                                      '_source': source_key,
                                      '_target': target_key,
                                      '_prob': prob,
                                      '_numeric': numeric,
                                      '_created': ts,
                                      '_updated': None,
                                      '_expired': None,
                                      '_changed': False}

            if len(edge_attributes) > 0:
                self.edges[triple_key].update(**edge_attributes)

            # add edge to nodes
            #
            self.nodes[source_key]['_edges'].add(triple_key)
            if not self.directional:
                self.nodes[target_key]['_edges'].add(triple_key)

        return triple_key

    def remove_edge(self,
                    triple: Union[TripleKey, TripleId],
                    audit: bool = False):

        if isinstance(triple, tuple):
            triple_id = triple
            triple_key = get_triple_key(triple_id=triple, directional=self.directional)
        else:
            triple_id = get_triple_id(triple_key=triple)
            triple_key = triple

        if triple_key in self.edges:

            if audit:
                ts = time.time()

                # construct an expired triple_id
                #
                expired_triple_id = (triple_id[0], (triple_id[1][0], triple_id[1][1], ts), triple_id[2])
                expired_edge_key = get_triple_key(triple_id=expired_triple_id, directional=self.directional)

                self.edges[expired_edge_key] = deepcopy(self.edges[triple_key])
                self.edges[expired_edge_key]['_changed'] = True
                self.edges[expired_edge_key]['_updated'] = ts
                self.edges[expired_edge_key]['_expired'] = ts

            # remove edge from nodes
            #
            self.nodes[self.edges[triple_key]['_source']]['_edges'].discard(triple_key)
            if not self.directional:
                self.nodes[self.edges[triple_key]['_target']]['_edges'].discard(triple_key)
 
            # delete the edge
            #
            del self.edges[triple_key]

            # delete edgeType entry if all edges have been removed
            #
            if not audit and sum([1 for t_key in self.edges if self.edges[t_key]['_type'] == triple_id[1][0]]) == 0:
                self.edgeTypes.discard(triple_id[1][0])

    def remove_node(self, node: Union[NodeKey, NodeId]):

        if isinstance(node, tuple):
            node_key = get_node_key(node_id=node)
        else:
            node_key = node

        if node_key in self.nodes:

            # first delete any edges node has
            #
            triple_keys = list(self.nodes[node_key]['_edges'])
            for triple_key in triple_keys:
                self.remove_edge(triple=triple_key, audit=False)

            del self.nodes[node_key]

            if node_key in self.embedded_graphs:
                del self.embedded_graphs[node_key]

    def compare_graph(self, graph_to_compare=None, compare_edge_types: Optional[Set[EdgeType]] = None) -> Tuple[float, Por]:

        distance: float = 0.0
        numeric_dist: float
        prob_dist: float

        por: Por = {}

        # if graph_to_compare is None then set to an empty graph
        #
        if graph_to_compare is None:
            graph_to_compare = AMGraph()

        if compare_edge_types is not None:

            # get the edges to compare - ie the edge type is in compare_edge_types and edge is not expired
            #
            triples_to_compare = ({triple_key
                                   for triple_key in self.edges
                                   if self.edges[triple_key]['_type'] in compare_edge_types and self.edges[triple_key]['_expired'] is None} |
                                  {triple_key
                                   for triple_key in graph_to_compare.edges
                                   if graph_to_compare.edges[triple_key]['_type'] in compare_edge_types and graph_to_compare.edges[triple_key]['_expired'] is None})
        else:
            triples_to_compare = ({triple_key
                                   for triple_key in self.edges
                                   if self.edges[triple_key]['_expired'] is None} |
                                  {triple_key
                                   for triple_key in graph_to_compare.edges
                                   if graph_to_compare.edges[triple_key]['_expired'] is None})

        for triple_key in triples_to_compare:

            # default numeric_dist in case edge numeric is None
            #
            numeric_dist = 0.0

            # if edge in both graphs
            #
            if triple_key in self.edges and triple_key in graph_to_compare.edges:
                prob_dist = abs(self.edges[triple_key]['_prob'] - graph_to_compare.edges[triple_key]['_prob'])

                if self.edges[triple_key]['_numeric'] is not None and graph_to_compare.edges[triple_key]['_numeric'] is not None:
                    numeric_dist = abs(self.edges[triple_key]['_numeric'] - graph_to_compare.edges[triple_key]['_numeric'])

            # if edge only in this graph
            #
            elif triple_key in self.edges:
                prob_dist = self.edges[triple_key]['_prob']

                if self.edges[triple_key]['_numeric'] is not None:
                    numeric_dist = self.edges[triple_key]['_numeric']

            # if edge only in graph_to_compare
            #
            else:
                prob_dist = graph_to_compare.edges[triple_key]['_prob']

                if graph_to_compare.edges[triple_key]['_numeric'] is not None:
                    numeric_dist = graph_to_compare.edges[triple_key]['_numeric']

            por[triple_key] = {'prob': prob_dist, 'numeric': numeric_dist}

            distance += pow(prob_dist, 2)

            distance += pow(numeric_dist, 2)

        distance = sqrt(distance)

        return distance, por

    def learn_graph(self, graph_to_learn=None, learn_rate: float = 1.0, learn_edge_types: Optional[Set[EdgeType]] = None, prune_threshold: float = 0.1, audit: bool = False):

        # if graph_to_lean is None then set to an empty graph
        #
        if graph_to_learn is None:
            graph_to_learn = AMGraph()

        if learn_edge_types is not None:

            # get the edges to compare - ie the edge type is in compare_edge_types and edge is not expired
            #
            exist_triples_to_learn = {triple_key
                                      for triple_key in self.edges
                                      if self.edges[triple_key]['_type'] in learn_edge_types and self.edges[triple_key]['_expired'] is None}
            triples_to_learn = (exist_triples_to_learn |
                                {triple_key
                                 for triple_key in graph_to_learn.edges
                                 if graph_to_learn.edges[triple_key]['_type'] in learn_edge_types and graph_to_learn.edges[triple_key]['_expired'] is None})
        else:
            exist_triples_to_learn = {triple_key
                                      for triple_key in self.edges
                                      if self.edges[triple_key]['_expired'] is None}
            triples_to_learn = (exist_triples_to_learn |
                                {triple_key
                                 for triple_key in graph_to_learn.edges
                                 if graph_to_learn.edges[triple_key]['_expired'] is None})

        # if no existing edges to learn then override learn rate to maximum (1.0)
        #
        if len(triples_to_learn) == 0:
            learn_rate = 1.0

        if audit:
            ts = time.time()
        else:
            ts = None

        triples_to_prune = set()

        for triple_key in triples_to_learn:

            # if edge in both graphs
            #
            if triple_key in self.edges and triple_key in graph_to_learn.edges:

                prob = self.edges[triple_key]['_prob'] + ((graph_to_learn.edges[triple_key]['_prob'] - self.edges[triple_key]['_prob']) * learn_rate)

                numeric = None

                if prob > prune_threshold:

                    if self.edges[triple_key]['_numeric'] is not None and graph_to_learn.edges[triple_key]['_numeric'] is not None:
                        numeric = self.edges[triple_key]['_numeric'] + ((graph_to_learn.edges[triple_key]['_numeric'] - self.edges[triple_key]['_numeric']) * learn_rate)

                    self.set_edge(triple=triple_key, audit=audit, timestamp=ts, numeric=numeric, prob=prob)
                else:
                    triples_to_prune.add(triple_key)

            # if edge only in this graph
            #
            elif triple_key in self.edges:

                prob = self.edges[triple_key]['_prob'] + ((0.0 - self.edges[triple_key]['_prob']) * learn_rate)

                numeric = None

                if prob > prune_threshold:

                    if self.edges[triple_key]['_numeric'] is not None:
                        numeric = self.edges[triple_key]['_numeric'] + ((0.0 - self.edges[triple_key]['_numeric']) * learn_rate)

                    self.set_edge(triple=triple_key, audit=audit, timestamp=ts, numeric=numeric, prob=prob)
                else:
                    triples_to_prune.add(triple_key)

            # if edge only in graph_to_learn
            #
            else:
                prob = (graph_to_learn.edges[triple_key]['_prob']) * learn_rate

                numeric = None

                if prob > prune_threshold:

                    if graph_to_learn.edges[triple_key]['_numeric'] is not None:
                        numeric = graph_to_learn.edges[triple_key]['_numeric'] * learn_rate

                    self.set_edge(triple=triple_key, audit=audit, timestamp=ts, numeric=numeric, prob=prob)

        # now copy over triples not learnt and not expired
        #
        if learn_edge_types is not None:
            triples_to_copy = {triple_key
                               for triple_key in graph_to_learn.edges
                               if graph_to_learn.edges[triple_key]['_type'] not in learn_edge_types and graph_to_learn.edges[triple_key]['_expired'] is None}

            for triple_key in triples_to_copy:

                prob = graph_to_learn.edges[triple_key]['_prob'] * learn_rate

                if graph_to_learn.edges[triple_key]['_numeric'] is not None:
                    numeric = graph_to_learn.edges[triple_key]['_numeric'] * learn_rate
                else:
                    numeric = None

                self.set_edge(triple=triple_key, audit=audit, timestamp=ts, numeric=numeric, prob=prob)

        # now prune triples
        #
        for triple_key in triples_to_prune:
            self.remove_edge(triple=triple_key, audit=audit)

    def learn_edge(self, triple: Union[TripleKey, TripleId], learn_rate, numeric: Optional[float] = None, prune_threshold: float = 0.0, audit: bool = False) -> TripleKey:

        if isinstance(triple, tuple):
            triple_id = triple
            triple_key = get_triple_key(triple_id=triple, directional=self.directional)
        else:
            triple_id = get_triple_id(triple_key=triple)
            triple_key = triple

        source_key = get_node_key(node_id=triple_id[0])
        if source_key not in self.nodes:
            self.set_edge(triple=triple, prob=1.0, numeric=numeric, audit=audit)
        else:
            triples_to_process = {existing_triple_key
                                  for existing_triple_key in self.nodes[source_key]['_edges']
                                  if self.edges[existing_triple_key]['_type'] == triple_id[1][0] and self.edges[existing_triple_key]['_expired'] is None}

            triples_to_process.add(triple_key)
            triples_to_prune = set()

            for triple_key_to_process in triples_to_process:
                if triple_key_to_process != triple_key:

                    # weaken the probability of this edge and reduce numeric closer to 0.0
                    #
                    existing_prob = self.edges[triple_key_to_process]['_prob'] + ((0.0 - self.edges[triple_key_to_process]['_prob']) * learn_rate)

                    if existing_prob > prune_threshold:
                        if self.edges[triple_key_to_process]['_numeric'] is not None:
                            existing_numeric = self.edges[triple_key_to_process]['_numeric'] + ((0.0 - self.edges[triple_key_to_process]['_numeric']) * learn_rate)
                        else:
                            existing_numeric = None

                        self.set_edge(triple=triple_key_to_process, prob=existing_prob, numeric=existing_numeric, audit=audit)

                    else:
                        triples_to_prune.add(triple_id)

                else:
                    if triple_key_to_process in self.edges:
                        new_prob = self.edges[triple_key_to_process]['_prob'] + ((1.0 - self.edges[triple_key_to_process]['_prob']) * learn_rate)
                        if numeric is not None and self.edges[triple_key_to_process]['_numeric'] is not None:
                            new_numeric = self.edges[triple_key_to_process]['_numeric'] + ((numeric - self.edges[triple_key_to_process]['_numeric']) * learn_rate)
                        else:
                            new_numeric = numeric
                    else:

                        # if there are more than 1 in triples_to_process this means edges for the correct type already exist and prob = learn_rate
                        #
                        if len(triples_to_process) > 1:
                            new_prob = learn_rate
                            if numeric is not None:
                                new_numeric = numeric * learn_rate
                            else:
                                new_numeric = None

                        # else probability needs to start from 1.0
                        #
                        else:
                            new_prob = 1.0
                            new_numeric = numeric

                    self.set_edge(triple=triple_key_to_process, prob=new_prob, numeric=new_numeric, audit=audit)
            # now prune triples
            #
            for triple_key in triples_to_prune:
                self.remove_edge(triple=triple_key, audit=audit)

        return triple_key

    def merge_graph(self, graph_to_merge, weight: float = 1.0, audit: bool = False):

        triples_to_merge = {triple_key
                            for triple_key in graph_to_merge.edges
                            if graph_to_merge.edges[triple_key]['_expired'] is None}

        if audit:
            ts = time.time()
        else:
            ts = None

        for triple_key in triples_to_merge:

            # if edge in both graphs
            #
            if triple_key in self.edges and triple_key in graph_to_merge.edges:

                prob = self.edges[triple_key]['_prob'] + (graph_to_merge.edges[triple_key]['_prob'] * weight)

                numeric = None

                if self.edges[triple_key]['_numeric'] is not None and graph_to_merge.edges[triple_key]['_numeric'] is not None:
                    numeric = self.edges[triple_key]['_numeric'] + (graph_to_merge.edges[triple_key]['_numeric'] * weight)

                self.set_edge(triple=triple_key, audit=audit, timestamp=ts, numeric=numeric, prob=prob)

            # if edge only in this graph
            #
            elif triple_key in graph_to_merge.edges:

                prob = (graph_to_merge.edges[triple_key]['_prob']) * weight

                numeric = None

                if graph_to_merge.edges[triple_key]['_numeric'] is not None:
                    numeric = graph_to_merge.edges[triple_key]['_numeric'] * weight

                self.set_edge(triple=triple_key, audit=audit, timestamp=ts, numeric=numeric, prob=prob)

    def diff_graph(self, graph_to_diff):

        for triple_key in graph_to_diff.edges:
            if triple_key in self.edges:
                if graph_to_diff.edges[triple_key]['_numeric'] is not None:
                    if self.edges[triple_key]['_numeric'] is not None:
                        self.edges[triple_key]['_numeric'] = self.edges[triple_key]['_numeric'] - graph_to_diff.edges[triple_key]['_numeric']
                    else:
                        self.edges[triple_key]['_numeric'] = -graph_to_diff.edges[triple_key]['_numeric']
            elif graph_to_diff.edges[triple_key]['_numeric'] is not None:
                self.set_edge(triple=triple_key, numeric=-graph_to_diff.edges[triple_key]['_numeric'])

    def rename_triples(self, postfix_edge_uid=None):

        for exist_triple_key in list(self.edges):

            exist_triple_id = get_triple_id(triple_key=exist_triple_key)
            new_triple = deepcopy(self.edges[exist_triple_key])

            new_triple['_uid'] = f'{new_triple["_uid"]}_{postfix_edge_uid}'
            new_triple_id = (exist_triple_id[0], (new_triple['_type'], new_triple['_uid'], new_triple['_expired']), exist_triple_id[2])
            new_triple_key = get_triple_key(triple_id=new_triple_id, directional=self.directional)

            del self.edges[exist_triple_key]
            self.edges[new_triple_key] = new_triple

            self.nodes[new_triple['_source']]['_edges'].discard(exist_triple_key)
            self.nodes[new_triple['_source']]['_edges'].add(new_triple_key)
            if not self.directional:
                self.nodes[new_triple['_target']]['_edges'].discard(exist_triple_key)
                self.nodes[new_triple['_target']]['_edges'].add(new_triple_key)

    def get_sub_graphs(self, generalise: bool = False):
        sub_graphs = []
        if self.directional:
            for node_key in self.nodes:
                if len(self.nodes[node_key]['_edges']) > 0:
                    graph = AMGraph(directional=True)

                    if generalise:
                        source_id = (self.nodes[node_key]['_type'], '*')
                    else:
                        source_id = (self.nodes[node_key]['_type'], self.nodes[node_key]['_uid'])

                    for edge_key in self.nodes[node_key]['_edges']:
                        edge_id = (self.edges[edge_key]['_type'], self.edges[edge_key]['_uid'], self.edges[edge_key]['_expired'])
                        target_id = (self.nodes[self.edges[edge_key]['_target']]['_type'], self.nodes[self.edges[edge_key]['_target']]['_uid'])
                        graph.set_edge(triple=(source_id, edge_id, target_id),
                                       prob=self.edges[edge_key]['_prob'],
                                       numeric=self.edges[edge_key]['_numeric'])
                    sub_graphs.append((self.nodes[node_key]['_type'], graph))

        return sub_graphs

    def calc_communities(self, community_edge_type: EdgeType, weight_field='_numeric', inverse=False):
        if len(self.edges) > 1:
            nx_graph = nx.MultiGraph()
            distances = [self.edges[triple_key][weight_field] for triple_key in self.edges if self.edges[triple_key]['_type'] == community_edge_type]
            min_distance = min(distances)
            max_distance = max(distances)
            for triple_key in self.edges:
                if self.edges[triple_key]['_type'] == community_edge_type and self.edges[triple_key]['_expired'] is None:
                    weight = (self.edges[triple_key][weight_field] - min_distance) / (max_distance - min_distance)
                    if inverse:
                        weight = 1 - weight
                    nx_graph.add_edge(self.edges[triple_key]['_source'], self.edges[triple_key]['_target'], weight=weight)

            communities = list(nx.algorithms.community.greedy_modularity_communities(nx_graph, weight='weight'))
            for c_idx in range(len(communities)):
                for node_key in communities[c_idx]:
                    self.nodes[node_key]['_community'] = c_idx


if __name__ == '__main__':

    from src.normalise_amgraph import NormaliseAMGraph

    g1 = AMGraph(directional=True)
    g1.set_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), numeric=200)

    g1.remove_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), audit=False)

    g1.remove_node(node=('A', '1'))

    g1.remove_node(node=('B', '2'))

    g1.set_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), numeric=200, audit=True)

    g1.set_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), muneric=400, audit=True)

    g1.remove_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), audit=True)

    g2 = AMGraph(directional=False)

    g2.set_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), numeric=200)

    g2.remove_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), audit=False)

    g2.remove_node(node=('A', '1'))

    g2.remove_node(node=('B', '2'))

    g2.set_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), numeric=200, audit=True)

    g2.set_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), numeric=400, audit=True)

    g2.remove_edge(triple=(('A', '1'), ('HAS_B', None, None), ('B', '2')), audit=True)

    g3 = AMGraph(directional=True)

    g3.set_edge(triple=(('TRADE', '*'), ('HAS_PLATFORM', None, None), ('PLATFORM', 'A')))
    g3.set_edge(triple=(('TRADE', '*'), ('HAS_DATE', None, None), ('DATE', '22-11-66')))
    g3.set_edge(triple=(('TRADE', '*'), ('HAS_VOLUME', None, None), ('VOLUME', 'TRADE')), numeric=100)

    g4 = AMGraph(directional=True)

    g4.set_edge(triple=(('TRADE', '*'), ('HAS_PLATFORM', None, None), ('PLATFORM', 'A')))
    g4.set_edge(triple=(('TRADE', '*'), ('HAS_DATE', None, None), ('DATE', '22-11-66')))
    g4.set_edge(triple=(('TRADE', '*'), ('HAS_VOLUME', None, None), ('VOLUME', 'TRADE')), numeric=200)

    normaliser = NormaliseAMGraph()
    g3n, new_min_max = normaliser.normalise(graph=g3)
    g4n, new_min_max = normaliser.normalise(graph=g4)

    distance = g3n.compare_graph(graph_to_compare=g4n)

    g3n1 = AMGraph(graph=g3n)

    g3n1.learn_graph(graph_to_learn=g4n, learn_rate=0.7)
    g3n1d = normaliser.denormalise(graph=g3n1)

    g3n1.learn_graph(graph_to_learn=g4n, learn_rate=0.7)
    g3n1d = normaliser.denormalise(graph=g3n1)

    g5 = AMGraph()
    g5.merge_graph(graph_to_merge=g3n, weight=0.5)
    g5d = normaliser.denormalise(graph=g5)

    g5.merge_graph(graph_to_merge=g4n, weight=0.5)
    g5d = normaliser.denormalise(graph=g5)

    g6 = AMGraph()
    g6.learn_edge(triple=(('NEURON', '1'), ('NN', None, None), ('NEURON', '2')), learn_rate=0.7)
    g6.learn_edge(triple=(('NEURON', '1'), ('NN', None, None), ('NEURON', '2')), learn_rate=0.7)
    g6.learn_edge(triple=(('NEURON', '1'), ('NN', None, None), ('NEURON', '3')), learn_rate=0.7)
    g6.learn_edge(triple=(('NEURON', '1'), ('NN', None, None), ('NEURON', '3')), learn_rate=0.7)

    g7 = AMGraph(directional=True)
    g7.set_edge(triple=(('TRADE', '1'), ('HAS_PLATFORM', None, None), ('PLATFORM', 'A')))
    g7.set_edge(triple=(('TRADE', '1'), ('HAS_DATE', None, None), ('DATE', '22-11-66')))
    g7.set_edge(triple=(('PLATFORM', 'A'), ('HAS_CHANNEL', None, None), ('CHANNEL', 'Electronic')))

    sub_graphs = g7.get_sub_graphs()

    sub_graphs_g = g7.get_sub_graphs(generalise=True)

    g8 = AMGraph()
    g8.set_node(node=('NEURON', '1'), a_graph=g3n1)

    jg8n = g8.to_dict()
    jg8dn = g8.to_dict(denormaliser=normaliser)

    g9 = AMGraph()
    g9.set_edge(triple=(('TRADE', '*'), ('HAS_VOLUME', None, None), ('VOLUME', 'TRADE')), numeric=200)

    stm = AMGraph(graph=g9)
    ltm = AMGraph(graph=g9)

    g10 = AMGraph()
    g10.set_edge(triple=(('TRADE', '*'), ('HAS_VOLUME', None, None), ('VOLUME', 'TRADE')), numeric=100)

    stm.learn_graph(graph_to_learn=g10, learn_rate=0.7)
    ltm.learn_graph(graph_to_learn=g10, learn_rate=0.4)

    dg = AMGraph(graph=stm)
    dg.diff_graph(graph_to_diff=ltm)
    dg.rename_triples(postfix_edge_uid='lstm')

    dg.merge_graph(graph_to_merge=g10, weight=1.0)


    print('finished')
