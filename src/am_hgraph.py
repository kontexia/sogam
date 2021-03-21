#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Tuple,  Set
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
    def get_node_key(node_type, node_uid) -> str:
        # ':' special - enables easy splitting of keys to create ids
        #
        return f'{node_type}:{node_uid}'

    @staticmethod
    def get_node_id(node_key: str) -> tuple:

        # ':' special - enables easy splitting of keys to create ids
        #
        node_id = tuple(node_key.split(':'))

        return node_id

    @staticmethod
    def get_edge_key(source_key, target_key, edge_type, directional: bool = False) -> str:

        # if not directional then key will consist of alphanumeric sort of source_key and target_key
        # note ':' special - delimits all components and allows for easy splitting to derive equivalent id
        #
        if (not directional and source_key < target_key) or directional:
            edge_key = f'{source_key}:{edge_type}:{target_key}'
        else:
            edge_key = f'{target_key}:{edge_type}:{source_key}'
        return edge_key

    @staticmethod
    def get_edge_id(edge_key: str) -> tuple:
        # assume special ':' character delimits components
        #
        edge_id = tuple(edge_key.split(':'))

        return edge_id

    def __init__(self, uid: Optional[str] = None, graph=None, directional=True, prune_threshold=0.01):

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

        # TODO
        # initialised from graph

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
            else:
                # auto generate uid
                #
                node_uid = f'{self.uid}:{AMHGraph.next_node_uid}'
                AMHGraph.next_node_uid += 1

        node_key = AMHGraph.get_node_key(node_type=node_type, node_uid=node_uid)

        if node_key not in self.nodes:

            if prob is None:
                prob = 1.0

            self.nodes[node_key] = {'_type': node_type,
                                    '_uid': node_uid,
                                    '_edges': set(),
                                    '_value': value,
                                    '_prob': prob,
                                    '_community': None,
                                    '_changed': False}

        else:
            # update the attributes that are set
            #
            self.nodes[node_key]['_changed'] = True
            if value is not None:
                self.nodes[node_key]['_value'] = value

            if prob is not None:
                self.nodes[node_key]['_prob'] = prob

        if len(node_attributes) > 0:
            self.nodes[node_key].update(**node_attributes)

        return node_key

    def set_edge(self,
                 source_key: NodeKey,
                 target_key: NodeKey,
                 edge_type: EdgeType,
                 prob: float = None,
                 **edge_attributes
                 ) -> EdgeKey:

        # can only add edges to existing nodes
        #
        if source_key in self.nodes and target_key in self.nodes:

            # create the edge key
            #
            edge_key = AMHGraph.get_edge_key(source_key=source_key, target_key=target_key, edge_type=edge_type, directional=self.directional)

            # if the edge already exists then just update the attributes
            #
            if edge_key in self.edges:

                self.edges[edge_key]['_changed'] = True

                if prob is not None:
                    self.edges[edge_key]['_prob'] = prob

                if len(edge_attributes) > 0:
                    self.edges[edge_key].updated(**edge_attributes)
            else:

                # if prob not set then default
                #
                if prob is None:
                    prob = 1.0

                self.edges[edge_key] = {'_type': edge_type,
                                        '_source': source_key,
                                        '_target': target_key,
                                        '_prob': prob,
                                        '_changed': False}

                if len(edge_attributes) > 0:
                    self.edges[edge_key].update(**edge_attributes)

                # add edge to nodes
                #
                self.nodes[source_key]['_edges'].add(edge_key)
                if not self.directional:
                    self.nodes[target_key]['_edges'].add(edge_key)
        else:
            edge_key = None
        return edge_key

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
                elif isinstance(self.nodes[node_key]['_value'], AMNumeric) and isinstance(graph_to_compare.nodes[node_key]['_value'], AMNumeric):
                    node_dist += abs(self.nodes[node_key]['_value'].value - graph_to_compare.nodes[node_key]['_value'].value)

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
                node_dist += 1.0

            # if the node is only in the graph_to_compare
            #
            else:
                # the difference in probability
                #
                node_dist += graph_to_compare.nodes[node_key]['_prob']

                # add max possible normalised difference for value comparison
                node_dist += 1.0

            # normalise the sum of the probability and value distances
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

    def learn(self, graph_to_learn=None, learn_rate: float=1.0, learn_node_types: Optional[Set[NodeType]] = None, learn_values=True):

        # if graph_to_compare is None then set to an empty graph
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
                        if isinstance(self.nodes[node_key]['_value'], AMNumeric):
                            self.nodes[node_key]['_value'].value += ((graph_to_learn.nodes[node_key]['_value'].value - self.nodes[node_key]['_value'].value) * learn_rate)

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

                    # learn values if required
                    #
                    if learn_values:

                        # if numeric then move value closer to zero
                        #
                        if isinstance(self.nodes[node_key]['_value'], AMNumeric):
                            self.nodes[node_key]['_value'].value -= (self.nodes[node_key]['_value'].value * learn_rate)

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
                    elif isinstance(graph_to_learn.nodes[node_key]['_value'], AMNumeric):
                        if learn_values:
                            self.nodes[node_key]['_value'] = AMNumeric(value=graph_to_learn.nodes[node_key]['_value'].value * learn_rate)
                        else:
                            self.nodes[node_key]['_value'] = graph_to_learn.nodes[node_key]['_value']

                    # if its a graph then either copy or take reference
                    #
                    else:
                        if learn_values:
                            new_graph = AMHGraph(uid=graph_to_learn.uid, directional=graph_to_learn.directional, prune_threshold=graph_to_learn.prune_threshold)
                            new_graph.learn(graph_to_learn=graph_to_learn, learn_rate=learn_rate, learn_node_types=learn_node_types, learn_values=learn_values)
                            self.nodes[node_key]['_value'] = new_graph
                        else:
                            self.nodes[node_key]['_value'] = graph_to_learn.nodes[node_key]['_value']

        # now prune nodes as required
        #
        for node_key in nodes_to_prune:
            for edge_key in self.nodes[node_key]['edges']:

                # get other node in connection
                #
                if self.edges[edge_key]['_source'] != node_key:
                    target_key = self.edges[edge_key]['_source']
                else:
                    target_key = self.edges[edge_key]['_target']

                # delete edge from other node
                #
                del self.nodes[target_key]['_edges'][edge_key]

                # delete edge from edges
                #
                del self.edges[edge_key]

            # delete node
            #
            del self.nodes[node_key]

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
                edge_prob = graph_to_learn.nodes[edge_key]['_prob'] * learn_rate

                # if probability of edge existing above threshold and the nodes exist then learn else prune
                #
                if edge_key > self.prune_threshold and graph_to_learn.nodes[edge_key]['_source'] in self.nodes and graph_to_learn.nodes[edge_key]['_target'] in self.nodes:

                    self.set_edge(source_key=graph_to_learn.nodes[edge_key]['_source'],
                                  target_key=graph_to_learn.nodes[edge_key]['_target'],
                                  edge_type=graph_to_learn[edge_key]['_type'],
                                  prob=edge_prob)

        # delete edges as required
        #
        for edge_key in edges_to_prune:

            # delete edge from nodes
            #
            source_key = self.edges[edge_key]['_source']
            target_key = self.edges[edge_key]['_target']
            del self.nodes[source_key]['_edges'][edge_key]
            del self.nodes[target_key]['_edges'][edge_key]

            # delete edge from edges
            #
            del self.edges[edge_key]


class AMNumeric(object):
    def __init__(self, value):
        self.value = value
        self.normalised = False


class MinMaxNormaliser(object):
    def __init__(self, type_groups=None):

        # map types to groups
        #
        self.types = {}

        # map groups to min, max and am_numerics
        #
        self.groups = {}
        if type_groups is not None:
            for atype in type_groups:
                self.types[atype] = type_groups[atype]

    def normalise(self, atype: str, numeric: AMNumeric) -> AMNumeric:

        if not numeric.normalised:
            if atype not in self.types:
                # default set group is also the type
                #
                group = atype
                self.types[atype] = group
                self.groups[group] = {'min': numeric.value - 0.01, 'max': numeric.value,
                                      'prev_min': numeric.value - 0.01, 'prev_max': numeric.value,
                                      'numerics': {numeric}}

            else:
                # get the group for the type
                #
                group = self.types[atype]

                if group not in self.groups:
                    self.groups[atype] = {'min': numeric.value - 0.01, 'max': numeric.value,
                                          'prev_min': numeric.value - 0.01, 'prev_max': numeric.value,
                                          'numerics': {numeric}}

                else:

                    # increase the max or decrease the min if required
                    #
                    renormalise = False
                    if numeric.value > self.groups[group]['max']:
                        self.groups[group]['prev_max'] = self.groups[group]['max']
                        self.groups[group]['max'] = numeric.value
                        renormalise = True
                    elif numeric.value < self.groups[group]['min']:
                        self.groups[group]['prev_min'] = self.groups[group]['min']
                        self.groups[group]['min'] = numeric.value
                        renormalise = True

                    if renormalise:
                        for exist_numeric in self.groups[group]['numerics']:
                            dn_value = (exist_numeric.value * (self.groups[group]['prev_max'] - self.groups[group]['prev_min'])) + self.groups[group]['prev_min']
                            exist_numeric.value = (dn_value - self.groups[group]['min']) / (self.groups[group]['max'] - self.groups[group]['min'])

            self.groups[atype]['numerics'].add(numeric)
            numeric.value = (numeric.value - self.groups[group]['min']) / (self.groups[group]['max'] - self.groups[group]['min'])
            numeric.normalised = True

        return numeric

    def denormalise(self, atype: str, numeric: AMNumeric) -> AMNumeric:

        if numeric.normalised:
            if atype in self.types:
                # get the group for the type
                #
                group = self.types[atype]

                numeric.value = (numeric.value * (self.groups[group]['max'] - self.groups[group]['min'])) + self.groups[group]['min']
                numeric.normalised = False

        return numeric

    def remove_numeric(self, atype, numeric):
        if atype in self.types:
            group = self.types[atype]
            if numeric in self.groups[group]['numerics']:
                del self.groups[group]['numerics'][numeric]


if __name__ == '__main__':

    normaliser = MinMaxNormaliser()

    g1 = AMHGraph()
    a1 = g1.set_node(node_type='A', node_uid='1', value=normaliser.normalise(atype='A', numeric=AMNumeric(100)))

    g2 = AMHGraph()
    g2.set_node(node_type='A', node_uid='1', value=normaliser.normalise(atype='A', numeric=AMNumeric(101)))

    por_1 = g1.compare_graph(graph_to_compare=g2)

    b1 = g1.set_node(node_type='B', value='Hello')
    g1.set_edge(source_key=a1, target_key=b1, edge_type='has_b')

    g2.set_node(node_type='B', value='Hello')
    g2.set_edge(source_key=a1, target_key=b1, edge_type='has_b')

    por_2 = g1.compare_graph(graph_to_compare=g2)

    g3 = AMHGraph(uid="composite_graph")
    ag = g3.set_node(node_type="AG", node_uid="a_graph", value=g1)

    g4 = AMHGraph(uid="composite_graph")
    g4.set_node(node_type="AG", node_uid="a_graph", value=g2)

    por_3 = g3.compare_graph(graph_to_compare=g4)

    e1 = g3.set_node(node_type='E', value='goodbye')
    g3.set_edge(source_key=ag, target_key=e1, edge_type='has_e')

    g4.set_node(node_type='E', value='goodbye')
    g4.set_edge(source_key=ag, target_key=e1, edge_type='has_e')

    por_4 = g3.compare_graph(graph_to_compare=g4)

    g1.learn(graph_to_learn=g2, learn_rate=0.7, learn_values=True)


    g3.learn(graph_to_learn=g4, learn_rate=0.7, learn_values=False)

    g3.learn(graph_to_learn=g4, learn_rate=0.7, learn_values=True)

    print('finished')