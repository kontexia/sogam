#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Tuple,  Set, Dict
from copy import deepcopy

# alias types
#
NodeType = str
""" the type of a node """

NodeUid = str
""" a node' unique identifier within the context of the node's type """

NodeId = Tuple[NodeType, NodeUid]
""" the unique identifier of a node - the concatenation of its' type and uid"""

NodeKey = str
""" json compliant unique identifier of a node used in hashes """

SourceId = NodeId
""" unique identifier of the source node in an edge """

TargetId = NodeId
""" unique identifier of the target node in an edge """

EdgeType = str
""" the type of an edge """

EdgeId = Tuple[SourceId, EdgeType, TargetId]
""" the unique identifier of an edge """

EdgeKey = str
""" json compliant unique identifier of an edge used in hashes """

Por = dict
""" dictionary representing the path of reasoning """


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class NoNodeFoundError(Error):
    """Exception raised when a Node is missing.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class NoEdgeFoundError(Error):
    """Exception raised when an Edge is missing.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class AMHGraph(object):

    # class attributes
    #

    # unique number for next graph
    #
    next_graph_uid = 0

    # unique number for next node
    #
    next_node_uid = 0

    @staticmethod
    def get_node_key(node_type, node_uid) -> NodeKey:
        # ':' special - enables easy splitting of keys to create ids
        #
        return f'{node_type}:{node_uid}'

    @staticmethod
    def get_node_id(node_key: NodeKey) -> NodeId:

        # ':' special - enables easy splitting of keys to create ids
        #
        split = node_key.split(':')
        node_id = (split[0], split[1])

        return node_id

    @staticmethod
    def get_edge_key(source_key, target_key, edge_type, directional: bool = False) -> EdgeKey:

        # if not directional then key will consist of alphanumeric sort of source_key and target_key
        # note ':' special - delimits all components and allows for easy splitting to derive equivalent id
        #
        if (not directional and source_key < target_key) or directional:
            edge_key = f'{source_key}:{edge_type}:{target_key}'
        else:
            edge_key = f'{target_key}:{edge_type}:{source_key}'
        return edge_key

    @staticmethod
    def get_edge_id(edge_key: EdgeKey) -> EdgeId:
        # assume special ':' character delimits components
        #
        split = edge_key.split(':')
        edge_id = ((split[0], split[1]), split[2], (split[3], split[4]))
        return edge_id

    def __init__(self, uid: Optional[str] = None, graph=None, directional: bool = True, prune_threshold: float = 0.01):

        # initialise from scratch
        #
        if graph is None:
            # every graph has a unique identifier
            #
            if uid is not None:
                self.uid = uid
            else:
                self.uid = f'_graph_{AMHGraph.next_graph_uid}'
                AMHGraph.next_graph_uid += 1

            # level below which a probability is considered equal to zero
            #
            self.prune_threshold = prune_threshold

            # dictionary of edges
            #
            self.edges: dict = {}

            # dictionary of nodes
            #
            self.nodes: dict = {}

            # If true then edges are directional
            #
            self.directional = directional

            # flag indicating if node numerics have been normalised
            #
            self.normalised = False

        # else initialise from either another AMHGraph of a dict
        #
        else:
            if isinstance(graph, AMHGraph):
                self.uid = graph.uid
                self.prune_threshold = graph.prune_threshold
                self.directional = graph.directional
                self.edges = deepcopy(graph.edges)
                self.nodes = deepcopy(graph.nodes)
                self.normalised = graph.normalised

            # else assume dict
            #
            else:
                self.from_dict(graph_dict=graph)

    def to_dict(self, denormaliser=None) -> dict:

        if denormaliser is None:
            graph = self
        else:
            graph = denormaliser.denormalise(graph=self, create_new=True)

        graph_dict = {'_type': 'AMHGraph',
                      '_uid': graph.uid,
                      '_prune_threshold': graph.prune_threshold,
                      '_directional': graph.directional,
                      '_normalised': graph.normalised,
                      '_nodes': deepcopy(graph.nodes),
                      '_edges': deepcopy(graph.edges)}

        # check if any nodes can contain graphs
        #
        for node_key in graph_dict['_nodes']:
            if isinstance(graph_dict['_nodes'][node_key]['_value'], AMHGraph):

                # convert to dict
                #
                graph_dict['_nodes'][node_key]['_value'] = graph_dict['_nodes'][node_key]['_value'].to_dict()

        return graph_dict

    def from_dict(self, graph_dict: dict):

        self.uid = graph_dict['_uid']
        self.directional = graph_dict['_directional']
        self.normalised = graph_dict['_normalised']
        self.prune_threshold = graph_dict['_prune_threshold']
        self.edges = deepcopy(graph_dict['_edges'])
        self.nodes = deepcopy(graph_dict['_nodes'])

        # convert node values into AMHGraphs if required
        #
        for node_key in self.nodes:

            # numerics and AMHGraphs represented as dicts so check
            #
            if isinstance(self.nodes[node_key]['_value'], dict) and self.nodes[node_key]['_value']['_type'] == 'AMHGraph':
                self.nodes[node_key]['_value'] = AMHGraph(graph=self.nodes[node_key]['_value'])

    def set_node(self,
                 node_type: NodeType,
                 value,
                 node_uid: NodeUid = None,
                 prob: float = None,
                 **node_attributes) -> NodeKey:

        if node_uid is None:
            # if the value is a string then its a valid uid
            #
            if isinstance(value, str):
                node_uid = value
            #else:
                # auto generate uid
                #
                #node_uid = f'{self.uid}:{AMHGraph.next_node_uid}'
                #AMHGraph.next_node_uid += 1

        node_key = AMHGraph.get_node_key(node_type=node_type, node_uid=node_uid)

        # if numeric convert to dictionary
        #
        if isinstance(value, float) or isinstance(value, int):
            value = {'_type': 'numeric', '_numeric': value}

        # add new node if it doesn't exist
        #
        if node_key not in self.nodes:

            # default to 1.0 if not provided
            #
            if prob is None:
                prob = 1.0

            self.nodes[node_key] = {'_type': node_type,
                                    '_uid': node_uid,
                                    '_edges': set(),
                                    '_value': value,
                                    '_prob': prob,
                                    '_community': None,
                                    '_changed': True}

        # else just update those attributes provided that are not None
        #
        else:
            self.nodes[node_key]['_changed'] = True
            if value is not None:
                self.nodes[node_key]['_value'] = value

            if prob is not None:
                self.nodes[node_key]['_prob'] = prob

        # assume any other attribute provided needs updating
        #
        if len(node_attributes) > 0:
            self.nodes[node_key].update(**node_attributes)

        return node_key

    def set_edge(self,
                 source_key: NodeKey,
                 target_key: NodeKey,
                 edge_type: EdgeType,
                 prob: float = None,
                 **edge_attributes) -> EdgeKey:

        # can only add edges to existing nodes
        #
        if source_key in self.nodes and target_key in self.nodes:

            # will need the edge key
            #
            edge_key = AMHGraph.get_edge_key(source_key=source_key, target_key=target_key, edge_type=edge_type, directional=self.directional)

            # if the edge already exists then just update the attributes that have changed
            #
            if edge_key in self.edges:

                self.edges[edge_key]['_changed'] = True

                if prob is not None:
                    self.edges[edge_key]['_prob'] = prob

                if len(edge_attributes) > 0:
                    self.edges[edge_key].updated(**edge_attributes)

            # add new edge
            #
            else:

                # if prob not set then default
                #
                if prob is None:
                    prob = 1.0

                self.edges[edge_key] = {'_type': edge_type,
                                        '_source': source_key,
                                        '_target': target_key,
                                        '_prob': prob,
                                        '_changed': True}

                if len(edge_attributes) > 0:
                    self.edges[edge_key].update(**edge_attributes)

                # add edge to nodes - if not directional then both nodes get a link
                #
                self.nodes[source_key]['_edges'].add(edge_key)
                if not self.directional:
                    self.nodes[target_key]['_edges'].add(edge_key)

        # the nodes don't exist so throw exception
        else:
            raise NoNodeFoundError('AMHGraph.set_edge', f"tried to create edge to {source_key} and {target_key} that don't exist")

        return edge_key

    def remove_node(self, node_key):

        if node_key in self.nodes:
            for edge_key in list(self.nodes[node_key]['_edges']):
                self.remove_edge(edge_key)

            # delete node
            #
            del self.nodes[node_key]
        else:
            raise NoNodeFoundError('AMHGraph.remove_node', f'{node_key} does not exist')

    def remove_edge(self, edge_key):

        if edge_key in self.edges:

            # delete edge from nodes
            #
            source_key = self.edges[edge_key]['_source']
            target_key = self.edges[edge_key]['_target']

            if source_key in self.nodes and edge_key in self.nodes[source_key]['_edges']:
                self.nodes[source_key]['_edges'].remove(edge_key)

            if target_key in self.nodes and edge_key in self.nodes[target_key]['_edges']:
                self.nodes[target_key]['_edges'].remove(edge_key)

            # delete edge from edges
            #
            del self.edges[edge_key]
        else:
            raise NoEdgeFoundError('AMHGraph.remove_edge', f'{edge_key} does not exist')

    def compare_graph(self, graph_to_compare=None, compare_node_types: Optional[Set[NodeType]] = None) -> Por:

        graph_distance = 0.0

        # keep track of the path of reasoning
        #
        por: Por = {'edges': {}, 'nodes': {}}

        # if graph_to_compare is None then set to an empty graph
        #
        if graph_to_compare is None:
            graph_to_compare = AMHGraph()

        # work out the nodes and edges that will be compared
        #
        if compare_node_types is not None:

            # get union of nodes that are of the right type
            #
            nodes_to_compare = ({node_key for node_key in self.nodes if self.nodes[node_key]['_type'] in compare_node_types} |
                                {node_key for node_key in graph_to_compare.nodes if graph_to_compare.nodes[node_key]['_type'] in compare_node_types})

            # only union of edges that are connected to nodes of the right type
            #
            edges_to_compare = ({edge_key
                                 for edge_key in self.edges
                                 if self.nodes[self.edges[edge_key]['_source']]['_type'] in compare_node_types and
                                 self.nodes[self.edges[edge_key]['_target']]['_type'] in compare_node_types} |
                                {edge_key
                                 for edge_key in graph_to_compare.edges
                                 if graph_to_compare.nodes[graph_to_compare.edges[edge_key]['_source']]['_type'] in compare_node_types and
                                 graph_to_compare.nodes[graph_to_compare.edges[edge_key]['_target']]['_type'] in compare_node_types})
        else:
            # all nodes and edges
            #
            nodes_to_compare = (self.nodes.keys()) | set(graph_to_compare.nodes.keys())
            edges_to_compare = (self.edges.keys()) | set(graph_to_compare.edges.keys())

        # compare nodes first
        #
        for node_key in nodes_to_compare:

            node_dist = 0.0

            # if node in both graphs
            #
            if node_key in self.nodes and node_key in graph_to_compare.nodes:

                # get the magnitude of the difference in probability
                #
                node_dist += abs(self.nodes[node_key]['_prob'] - graph_to_compare.nodes[node_key]['_prob'])

                # compare strings
                #
                if isinstance(self.nodes[node_key]['_value'], str) and isinstance(graph_to_compare.nodes[node_key]['_value'], str):
                    # if strings not identical then distance is 1.0
                    #
                    if self.nodes[node_key]['_value'] != graph_to_compare.nodes[node_key]['_value']:
                        node_dist += 1.0

                # compare numerics
                #
                elif isinstance(self.nodes[node_key]['_value'], dict) and isinstance(graph_to_compare.nodes[node_key]['_value'], dict):
                    node_dist += abs(self.nodes[node_key]['_value']['_numeric'] - graph_to_compare.nodes[node_key]['_value']['_numeric'])

                # compare graphs
                #
                elif isinstance(self.nodes[node_key]['_value'], AMHGraph) and isinstance(graph_to_compare.nodes[node_key]['_value'], AMHGraph):
                    n_por = self.nodes[node_key]['_value'].compare_graph(graph_to_compare=graph_to_compare.nodes[node_key]['_value'],
                                                                         compare_node_types=compare_node_types)
                    # get out the normalised distance between the two graphs
                    #
                    node_dist += n_por['graph']['norm']

                # else add max normalised distance
                else:
                    node_dist += 1.0

            # if the node is only in this graph then assume graph_to_compare has zero for node probability
            #
            elif node_key in self.nodes:

                # the difference in probability
                #
                node_dist += self.nodes[node_key]['_prob']

                # add max possible normalised difference for value comparison
                #
                node_dist += 1.0

            # if the node is only in the graph_to_compare
            #
            else:
                # the difference in probability
                #
                node_dist += graph_to_compare.nodes[node_key]['_prob']

                # add max possible normalised difference for value comparison
                #
                node_dist += 1.0

            # normalise the difference a node can be to be between 0 and 1
            #
            node_dist /= 2.0

            # update the path of reasoning
            #
            por['nodes'][node_key] = node_dist

            # keep track of the sum
            #
            graph_distance += node_dist

        # now compare edges
        #
        for edge_key in edges_to_compare:

            # if edge in both graphs calc distance between probabilities
            #
            if edge_key in self.edges and edge_key in graph_to_compare.edges:
                edge_dist = abs(self.edges[edge_key]['_prob'] - graph_to_compare.edges[edge_key]['_prob'])

            # if edge only in this graph
            #
            elif edge_key in self.edges:
                edge_dist = self.edges[edge_key]['_prob']

            # if edge only in graph_to_compare
            #
            else:
                edge_dist = graph_to_compare.edges[edge_key]['_prob']

            por['edges'][edge_key] = edge_dist

            graph_distance += edge_dist

        # normalise the distance by the number of unique nodes and edges
        #
        if len(nodes_to_compare) + len(edges_to_compare) > 0:
            norm_dist = graph_distance / (len(nodes_to_compare) + len(edges_to_compare))
        else:
            norm_dist = graph_distance

        por['graph'] = {'actual': graph_distance, 'norm': norm_dist}

        return por

    def learn(self, graph_to_learn=None, learn_rate: float = 1.0, learn_node_types: Optional[Set[NodeType]] = None, learn_values=True):

        # if graph_to_learn is None then set to an empty graph
        #
        if graph_to_learn is None:
            graph_to_learn = AMHGraph()

        # work out the nodes that will be learnt
        #
        if learn_node_types is not None:

            # get union of nodes that are of the right type
            #
            nodes_to_learn = ({node_key for node_key in self.nodes if self.nodes[node_key]['_type'] in learn_node_types} |
                              {node_key for node_key in graph_to_learn.nodes if graph_to_learn.nodes[node_key]['_type'] in learn_node_types})

        else:
            # all nodes
            #
            nodes_to_learn = (self.nodes.keys()) | set(graph_to_learn.nodes.keys())

        # if this graph has no edges and nodes then ensure learn_rate is 1.0
        # which will copy over from graph_to_learn
        #
        if (len(self.edges) + len(self.nodes)) == 0:
            learn_rate = 1.0

        # keep track of the nodes that need to be pruned
        #
        nodes_to_prune = set()

        for node_key in nodes_to_learn:

            # if node in both graphs
            #
            if node_key in self.nodes and node_key in graph_to_learn.nodes:
                prob = self.nodes[node_key]['_prob'] + ((graph_to_learn.nodes[node_key]['_prob'] - self.nodes[node_key]['_prob']) * learn_rate)

                # learn if probability of node existing is above threshold
                #
                if prob > self.prune_threshold:

                    self.nodes[node_key]['_prob'] = prob

                    # learn value if required
                    #
                    if learn_values:

                        # if numeric move the numeric closed to graph_to_learn numeric
                        #
                        if isinstance(self.nodes[node_key]['_value'], dict):
                            self.nodes[node_key]['_value']['_numeric'] += ((graph_to_learn.nodes[node_key]['_value']['_numeric'] - self.nodes[node_key]['_value']['_numeric']) *
                                                                           learn_rate)

                        # if a graph then learn the graph
                        #
                        elif isinstance(self.nodes[node_key]['_value'], AMHGraph):
                            self.nodes[node_key]['_value'].learn(graph_to_learn=graph_to_learn.nodes[node_key]['_value'],
                                                                 learn_rate=learn_rate,
                                                                 learn_node_types=learn_node_types,
                                                                 learn_values=learn_values)
                # else prune the node
                # later
                else:
                    nodes_to_prune.add(node_key)

            # node only in this graph
            #
            elif node_key in self.nodes:

                # probability should be reduced as missing from graph_to_learn
                #
                prob = self.nodes[node_key]['_prob'] - (self.nodes[node_key]['_prob'] * learn_rate)

                # learn if probability of this node existing is above threshold
                #
                if prob > self.prune_threshold:

                    self.nodes[node_key]['_prob'] = prob

                    # learn values if required
                    #
                    if learn_values:

                        # if numeric then move value closer to zero
                        #
                        if isinstance(self.nodes[node_key]['_value'], dict):
                            self.nodes[node_key]['_value']['_numeric'] -= (self.nodes[node_key]['_value']['_numeric'] * learn_rate)

                        # if graph then learn an empty graph
                        #
                        elif isinstance(self.nodes[node_key]['_value'], AMHGraph):
                            self.nodes[node_key]['_value'].learn(graph_to_learn=None,
                                                                 learn_rate=learn_rate,
                                                                 learn_node_types=learn_node_types,
                                                                 learn_values=learn_values)

                # else prune node
                # later
                else:
                    nodes_to_prune.add(node_key)

            # node only in graph_to_learn
            #
            else:
                node_prob = graph_to_learn.nodes[node_key]['_prob'] * learn_rate

                # if probability of this node existing above threshold then learn
                #
                if node_prob > self.prune_threshold:

                    # first create new node with no value
                    #
                    self.set_node(node_type=graph_to_learn.nodes[node_key]['_type'],
                                  node_uid=graph_to_learn.nodes[node_key]['_uid'],
                                  prob=node_prob,
                                  value=None)

                    # if value is a string then need to copy
                    #
                    if isinstance(graph_to_learn.nodes[node_key]['_value'], str):
                        self.nodes[node_key]['_value'] = graph_to_learn.nodes[node_key]['_value']

                    # if its a numeric then either copy or take reference
                    #
                    elif isinstance(graph_to_learn.nodes[node_key]['_value'], dict):
                        if learn_values:
                            self.nodes[node_key]['_value']['_numeric'] = graph_to_learn.nodes[node_key]['_value']['_numeric'] * learn_rate
                        else:
                            # just take reference
                            #
                            self.nodes[node_key]['_value'] = graph_to_learn.nodes[node_key]['_value']

                    # if its a graph then either copy or take reference
                    #
                    else:
                        if learn_values:
                            new_graph = AMHGraph(uid=graph_to_learn.uid, directional=graph_to_learn.directional, prune_threshold=graph_to_learn.prune_threshold)
                            new_graph.learn(graph_to_learn=graph_to_learn, learn_rate=learn_rate, learn_node_types=learn_node_types, learn_values=learn_values)
                            self.nodes[node_key]['_value'] = new_graph
                        else:
                            # just take reference
                            #
                            self.nodes[node_key]['_value'] = graph_to_learn.nodes[node_key]['_value']

        # now prune nodes as required
        #
        for node_key in nodes_to_prune:
            self.remove_node(node_key=node_key)

        # TODO
        # do we need to worry about copying over the nodes that are not learnt?

        # work out the edges that will be learnt
        #
        if learn_node_types is not None:

            # only union of edges that are connected to nodes of the right type
            #
            edges_to_learn = ({edge_key
                               for edge_key in self.edges
                               if self.nodes[self.edges[edge_key]['_source']]['_type'] in learn_node_types and
                               self.nodes[self.edges[edge_key]['_target']]['_type'] in learn_node_types} |
                              {edge_key
                               for edge_key in graph_to_learn.edges
                               if graph_to_learn.nodes[graph_to_learn.edges[edge_key]['_source']]['_type'] in learn_node_types and
                               graph_to_learn.nodes[graph_to_learn.edges[edge_key]['_target']]['_type'] in learn_node_types})
        else:
            # will learn all edges
            #
            edges_to_learn = (self.edges.keys()) | set(graph_to_learn.edges.keys())

        # keep track of  edges that can be pruned
        #
        edges_to_prune = set()

        for edge_key in edges_to_learn:

            # if edge in both graphs
            #
            if edge_key in self.edges and edge_key in graph_to_learn.edges:

                prob = self.edges[edge_key]['_prob'] + ((graph_to_learn.edges[edge_key]['_prob'] - self.edges[edge_key]['_prob']) * learn_rate)

                # if probability of edge existing above threshold then learn else prune
                #
                if prob > self.prune_threshold:
                    self.edges[edge_key]['_prob'] = prob
                else:
                    edges_to_prune.add(edge_key)

            # edge only in this graph
            #
            elif edge_key in self.edges:

                # probability should be reduced as missing from graph_to_learn
                #
                prob = self.edges[edge_key]['_prob'] - (self.edges[edge_key]['_prob'] * learn_rate)

                # if probability of edge existing above threshold then learn else prune
                #
                if prob > self.prune_threshold:
                    self.edges[edge_key]['_prob'] = prob
                else:
                    edges_to_prune.add(edge_key)

            # edge only in graph_to_learn
            #
            else:
                edge_prob = graph_to_learn.edges[edge_key]['_prob'] * learn_rate

                # if probability of edge existing above threshold and the nodes exist then learn else prune
                #
                if edge_prob > self.prune_threshold and graph_to_learn.edges[edge_key]['_source'] in self.nodes and graph_to_learn.edges[edge_key]['_target'] in self.nodes:

                    self.set_edge(source_key=graph_to_learn.edges[edge_key]['_source'],
                                  target_key=graph_to_learn.edges[edge_key]['_target'],
                                  edge_type=graph_to_learn.edges[edge_key]['_type'],
                                  prob=edge_prob)

        # TODO
        # do we need to worry about copying over the edges that are not learnt?

        # delete edges as required
        #
        for edge_key in edges_to_prune:
            self.remove_edge(edge_key=edge_key)

    def merge_graph(self, graph_to_merge, weight=1.0):

        for node_key in graph_to_merge.nodes.keys():

            # if node in both graphs
            #
            if node_key in self.nodes:

                self.nodes[node_key]['_prob'] += (graph_to_merge.nodes[node_key]['_prob'] * weight)

                # if numeric then add
                #
                if isinstance(self.nodes[node_key]['_value'], dict):
                    self.nodes[node_key]['_value']['_numeric'] += (graph_to_merge.nodes[node_key]['_value']['_numeric'] * weight)

                # if graph then merge
                #
                elif isinstance(self.nodes[node_key]['_value'], AMHGraph):
                    self.nodes[node_key]['_value'].merge_graph(graph_to_merge=graph_to_merge.nodes[node_key]['_value']['_numeric'],
                                                               weight=weight)

            # node only in graph_to_learn
            #
            else:

                # if its a numeric
                #
                if isinstance(graph_to_merge.nodes[node_key]['_value'], dict):
                    node_value = graph_to_merge.nodes[node_key]['_value']['_numeric'] * weight

                # if it's a graph
                #
                elif isinstance(graph_to_merge.nodes[node_key]['_value'], AMHGraph):
                    node_value = AMHGraph(uid=graph_to_merge.nodes[node_key]['_value'].uid)
                    node_value.merge_graph(graph_to_merge=graph_to_merge.nodes[node_key]['_value'], weight=weight)

                # else must be a string
                #
                else:
                    node_value = graph_to_merge.nodes[node_key]['_value']
                self.set_node(node_type=graph_to_merge.nodes[node_key]['_type'],
                              node_uid=graph_to_merge.nodes[node_key]['_uid'],
                              value=node_value,
                              prob=graph_to_merge.nodes[node_key]['_prob'] * weight)

        for edge_key in graph_to_merge.edges.keys():

            # edge in this graph
            #
            if edge_key in self.edges:

                self.edges[edge_key]['_prob'] += graph_to_merge.edges[edge_key]['_prob'] * weight

            # edge only in graph_to_learn
            #
            else:
                self.set_edge(source_key=graph_to_merge.edges[edge_key]['_source'],
                              target_key=graph_to_merge.edges[edge_key]['_target'],
                              edge_type=graph_to_merge.edges[edge_key]['_type'],
                              prob=graph_to_merge.edges[edge_key]['_prob'] * weight)


class GraphNormaliser(object):
    def __init__(self, normaliser=None):

        if normaliser is None:
            # map types to groups
            #
            self.types = {}

            # map groups to min, max and graphs
            #
            self.groups = {}

        elif isinstance(normaliser, dict):

            self.types = normaliser['_types']
            self.groups = normaliser['_groups']
        elif isinstance(normaliser, GraphNormaliser):
            self.types = deepcopy(normaliser.types)
            self.groups = deepcopy(normaliser.groups)

    def to_dict(self):
        gn_dict = {'_types': deepcopy(self.types),
                   '_groups': deepcopy(self.groups)}
        return gn_dict

    def normalise(self, graph: AMHGraph, create_new: bool = True) -> Tuple[AMHGraph, bool]:

        if create_new:
            # create a new graph that will hold the normalised values
            #
            n_graph = AMHGraph(graph=graph)
        else:
            n_graph = graph

        renormalise = False

        if not n_graph.normalised:
            for node_key in n_graph.nodes:
                if isinstance(n_graph.nodes[node_key]['_value'], dict):
                    a_type = n_graph.nodes[node_key]['_type']
                    if a_type not in self.types:

                        # default set group is also the type
                        #
                        group = a_type
                        self.types[a_type] = group
                        self.groups[group] = {'min': n_graph.nodes[node_key]['_value']['_numeric'], 'max': n_graph.nodes[node_key]['_value']['_numeric'],
                                              'prev_min': n_graph.nodes[node_key]['_value']['_numeric'], 'prev_max': n_graph.nodes[node_key]['_value']['_numeric']}

                    else:
                        # get the group for the type
                        #
                        group = self.types[a_type]

                        if group not in self.groups:
                            self.groups[a_type] = {'min': n_graph.nodes[node_key]['_value']['_numeric'], 'max': n_graph.nodes[node_key]['_value']['_numeric'],
                                                   'prev_min': n_graph.nodes[node_key]['_value']['_numeric'], 'prev_max': n_graph.nodes[node_key]['_value']['_numeric']}

                        else:

                            # increase the max or decrease the min if required
                            #
                            if n_graph.nodes[node_key]['_value']['_numeric'] > self.groups[group]['max']:
                                self.groups[group]['prev_max'] = self.groups[group]['max']
                                self.groups[group]['max'] = n_graph.nodes[node_key]['_value']['_numeric']
                                renormalise = True
                            elif n_graph.nodes[node_key]['_value']['_numeric'] < self.groups[group]['min']:
                                self.groups[group]['prev_min'] = self.groups[group]['min']
                                self.groups[group]['min'] = n_graph.nodes[node_key]['_value']['_numeric']
                                renormalise = True

                    if self.groups[group]['max'] - self.groups[group]['min'] > 0.0:
                        n_graph.nodes[node_key]['_value']['_numeric'] = ((n_graph.nodes[node_key]['_value']['_numeric'] - self.groups[group]['min']) /
                                                                         (self.groups[group]['max'] - self.groups[group]['min']))
                    else:
                        n_graph.nodes[node_key]['_value']['_numeric'] = 1.0
                    n_graph.normalised = True

        return n_graph, renormalise

    def denormalise(self, graph: AMHGraph, create_new: bool = True, prev_min_max=False) -> AMHGraph:

        if graph.normalised:

            if create_new:
                # create a new graph that will hold the normalised values
                #
                dn_graph = AMHGraph(graph=graph)
            else:
                dn_graph = graph

            for node_key in dn_graph.nodes:
                if isinstance(dn_graph.nodes[node_key]['_value'], dict):
                    a_type = dn_graph.nodes[node_key]['_type']

                    if a_type in self.types:

                        # get the group for the type
                        #
                        group = self.types[a_type]

                        if prev_min_max:
                            dn_graph.nodes[node_key]['_value']['_numeric'] = ((dn_graph.nodes[node_key]['_value']['_numeric'] * (self.groups[group]['prev_max'] - self.groups[group]['prev_min'])) +
                                                                              self.groups[group]['prev_min'])
                        else:
                            dn_graph.nodes[node_key]['_value']['_numeric'] = ((dn_graph.nodes[node_key]['_value']['_numeric'] * (self.groups[group]['max'] - self.groups[group]['min'])) +
                                                                              self.groups[group]['min'])
                        dn_graph.normalised = False

                # if node contains graph then denormalise it
                #
                elif isinstance(dn_graph.nodes[node_key]['_value'], AMHGraph):
                    dn_graph.nodes[node_key]['_value'] = self.denormalise(graph=dn_graph.nodes[node_key]['_value'], create_new=False, prev_min_max=prev_min_max)
        else:
            dn_graph = graph
        return dn_graph

    def renormalise(self, graph: AMHGraph, create_new: bool = True) -> AMHGraph:
        dn_graph = self.denormalise(graph=graph, create_new=create_new, prev_min_max=True)

        n_graph, _ = self.normalise(graph=dn_graph, create_new=False)

        return n_graph


if __name__ == '__main__':

    normaliser = GraphNormaliser()

    g1 = AMHGraph()
    a1 = g1.set_node(node_type='A', node_uid='1', value=100)
    g1, _ = normaliser.normalise(graph=g1, create_new=False)

    g2 = AMHGraph()
    g2.set_node(node_type='A', node_uid='1', value=101)
    g2, renormalise = normaliser.normalise(graph=g2, create_new=False)

    if renormalise:
        g1 = normaliser.renormalise(graph=g1, create_new=False)

    por_1 = g1.compare_graph(graph_to_compare=g2)

    # add another edge
    #
    b1 = g1.set_node(node_type='B', value='Hello')
    g1.set_edge(source_key=a1, target_key=b1, edge_type='has_b')

    g2.set_node(node_type='B', value='Hello')
    g2.set_edge(source_key=a1, target_key=b1, edge_type='has_b')

    por_2 = g1.compare_graph(graph_to_compare=g2)

    # composite graphs containing g1 and g2
    #
    g3 = AMHGraph(uid="composite_graph")
    ag = g3.set_node(node_type="AG", node_uid="a_graph", value=g1)

    g4 = AMHGraph(uid="composite_graph")
    g4.set_node(node_type="AG", node_uid="a_graph", value=g2)

    por_3 = g3.compare_graph(graph_to_compare=g4)

    # add node and edges to composite graphs
    #
    e1 = g3.set_node(node_type='E', value='hello')
    g3.set_edge(source_key=ag, target_key=e1, edge_type='has_e')

    e2 = g4.set_node(node_type='E', value='goodbye')
    g4.set_edge(source_key=ag, target_key=e2, edge_type='has_e')

    por_4 = g3.compare_graph(graph_to_compare=g4)

    # learn standard graph
    #
    g1.learn(graph_to_learn=g2, learn_rate=0.7, learn_values=True)

    # learn composite graph
    #
    g3.learn(graph_to_learn=g4, learn_rate=0.7, learn_values=False)


    g3.learn(graph_to_learn=g4, learn_rate=0.7, learn_values=True)

    print('finished')