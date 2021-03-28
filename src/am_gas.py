#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.am_hgraph import AMHGraph, NodeType, Por, NodeId, NodeKey, GraphNormaliser
from typing import Optional, Set, Tuple


NeuronId = NodeId
""" unique identifier for a neuron within the gas """

NeuronKey = NodeKey
""" json compatible unique identifier of neuron within the gas"""


class AMGas(object):
    def __init__(self,
                 fabric_name: str,
                 domain: str,
                 anomaly_threshold_factor: float = 4.0,
                 fast_alpha: float = 0.7,
                 prune_threshold: float = 0.01,
                 prune_nodes: bool = False,
                 normaliser: Optional[GraphNormaliser] = None,
                 delete_old_neurons: bool = False):

        self.fabric_name: str = fabric_name
        self.domain: str = domain
        self.anomaly_threshold_factor: float = anomaly_threshold_factor
        self.fast_alpha = fast_alpha
        self.slow_alpha = 1 - fast_alpha
        self.anomaly_threshold: float = 0.0
        self.anomalies: dict = {}
        self.prune_threshold: float = prune_threshold
        self.neural_gas: AMHGraph = AMHGraph(directional=False)

        # neural gas by default should be normalised
        #
        self.neural_gas.normalised = True

        self.update_id: int = 0
        self.next_neuron_id: int = 0
        self.last_bmu_key: Optional[NeuronKey] = None
        self.ema_error: Optional[float] = None
        self.ema_variance: float = 0.0
        self.motif_threshold: float = 0.0
        self.motifs: dict = {}
        self.search_node_types: set = set()
        self.prune_nodes: bool = prune_nodes
        self.updated: bool = True
        self.delete_old_neurons = delete_old_neurons

        if normaliser is None:
            self.normaliser = GraphNormaliser()
        else:
            self.normaliser = normaliser

    def to_dict(self, denormalise: bool = False) -> dict:
        d_gas = {'fabric_name': self.fabric_name,
                 'domain': self.domain,
                 'fast_alpha': self.fast_alpha,
                 'slow_alpha': self.slow_alpha,
                 'prune_threshold': self.prune_threshold,
                 'update_id': self.update_id,
                 'next_neuron_id': self.next_neuron_id,
                 'last_bmu_key': self.last_bmu_key,
                 'ema_error': self.ema_error,
                 'ema_variance': self.ema_variance,
                 'anomaly_threshold_factor': self.anomaly_threshold_factor,
                 'anomaly_threshold': self.anomaly_threshold,
                 'anomalies': self.anomalies,
                 'motif_threshold': self.motif_threshold,
                 'motifs': self.motifs,
                 'search_node_types': self.search_node_types,
                 'prune_nodes': self.prune_nodes,
                 'normaliser': None,
                 'delete_old_neurons': self.delete_old_neurons
                 }
        if self.normaliser is not None:
            d_gas['normaliser'] = self.normaliser.to_dict()
            if denormalise:
                d_gas['neural_gas'] = self.normaliser.denormalise(graph=self.neural_gas).to_dict(denormaliser=self.normaliser)
            else:
                d_gas['neural_gas'] = self.neural_gas.to_dict()
        else:
            d_gas['neural_gas'] = self.neural_gas.to_dict()

        return d_gas

    def from_dict(self, d_gas: dict, normalise: bool = False):
        self.fabric_name = d_gas['fabric_name']
        self.domain = d_gas['domain']
        self.fast_alpha = d_gas['fast_alpha']
        self.slow_alpha = d_gas['slow_alpha']
        self.prune_threshold = d_gas['prune_threshold']
        self.next_neuron_id = d_gas['next_neuron_id']
        self.last_bmu_key = d_gas['last_bmu_key']
        self.ema_error = d_gas['ema_error']
        self.ema_variance = d_gas['ema_variance']
        self.anomaly_threshold_factor = d_gas['anomaly_threshold_factor']
        self.anomaly_threshold = d_gas['anomaly_threshold']
        self.anomalies = d_gas['anomalies']
        self.motif_threshold = d_gas['motif_threshold']
        self.motifs = d_gas['motifs']
        self.search_node_types = d_gas['search_node_types']
        self.prune_nodes = d_gas['prune_nodes']
        self.delete_old_neurons = d_gas['delete_old_neurons']
        if d_gas['normaliser'] is not None:
            self.normaliser = GraphNormaliser(normaliser=d_gas['normaliser'])
            if normalise:
                self.neural_gas = self.normaliser.normalise(graph=AMHGraph(directional=False, graph=d_gas['neural_gas']))
            else:
                self.neural_gas = AMHGraph(directional=False, graph=d_gas['neural_gas'])
        else:
            self.normaliser = None
            self.neural_gas = AMHGraph(directional=False, graph=d_gas['neural_gas'])

    def update_gas_error(self, bmu_key: NeuronKey, bmu_distance: float, ref_id: str) -> Tuple[bool, bool]:

        # update the ema error and variance using the slow_alpha
        #
        if self.ema_error is None:
            self.ema_error = bmu_distance
        else:
            self.ema_error += (bmu_distance - self.ema_error) * self.slow_alpha

        self.ema_variance += (pow((bmu_distance - self.ema_error), 2) - self.ema_variance) * self.slow_alpha

        # record breaches of anomaly threshold
        #
        report: dict = {'bmu_key': bmu_key, 'mapped': self.update_id, 'error': bmu_distance, 'ref_id': ref_id}
        anomaly = False
        motif = False

        # determine if anomaly or motif detected
        #
        if bmu_distance > self.anomaly_threshold:
            self.anomalies[str(self.update_id)] = report
            anomaly = True
        elif self.motif_threshold is not None and bmu_distance <= self.motif_threshold:
            self.motifs[str(self.update_id)] = report
            motif = True

        # update threshold for next training data
        #
        stdev = pow(self.ema_variance, 0.5)
        self.anomaly_threshold = self.ema_error + (self.anomaly_threshold_factor * stdev)
        self.motif_threshold = max(self.ema_error - (2.0 * stdev), 0.0)

        return anomaly, motif

    # TODO
    def calc_communities(self):
        self.neural_gas.calc_communities(community_edge_type='NN', weight_field='_numeric', inverse=True)

    def add_neuron(self, graph: AMHGraph, distance_threshold: float = 0.0) -> NeuronKey:

        neuron_id = f'{self.next_neuron_id}'
        self.next_neuron_id += 1

        update_id = str(self.update_id)

        # make sure the graph to generalise has the prune threshold set
        #
        graph.prune_threshold = self.prune_threshold

        # a neuron is just a node in the neural gas graph
        #
        neuron_key = self.neural_gas.set_node(node_type='NEURON',
                                              node_uid=neuron_id,

                                              # NOTE: a neuron generalises a subgraph
                                              #
                                              value=graph,
                                              domain=self.domain,
                                              update_id=update_id,
                                              threshold=distance_threshold,
                                              ema_error=None,
                                              n_bmu=1,
                                              last_bmu=update_id,
                                              n_runner_up=0,
                                              last_runner_up=None,
                                              activation=1.0,

                                              # set the learning rate for the next update
                                              #
                                              learn_rate=self.fast_alpha
                                              )
        self.last_bmu_key = neuron_key
        return neuron_key

    def train(self,
              training_graph: AMHGraph,
              ref_id: str,
              search_node_types: Set[NodeType],
              learn_node_types: Set[NodeType]) -> Por:

        por: Por = {'fabric': self.fabric_name,
                    'domain': self.domain,
                    'ref_id': ref_id,
                    'bmu_key': None,
                    'bmu_distance': None,
                    'bmu_distance_threshold': 0.0,
                    'new_neuron_key': None,
                    'nn_neurons': [],
                    'anomaly': False,
                    'motif': False,
                    'ema_error': self.ema_error,
                    'ema_variance': self.ema_variance,
                    'anomaly_threshold': self.anomaly_threshold,
                    'motif_threshold': self.motif_threshold,
                    'deleted_neuron_key': None
                    }

        self.update_id += 1
        self.updated = True

        self.search_node_types.update(search_node_types)

        if self.normaliser is not None:
            t_graph, renormalise = self.normaliser.normalise(graph=training_graph)
        else:
            t_graph = AMHGraph(graph=training_graph)
            renormalise = False

        # if there are no neurons then add first one
        #
        if len(self.neural_gas.nodes) == 0:

            # neuron will have a distance threshold of 0.0 which will force the next training data point to be represented in another neuron
            #
            new_neuron_key = self.add_neuron(graph=t_graph, distance_threshold=0.0)
            por['new_neuron_key'] = new_neuron_key

        else:

            if renormalise:

                # renormalise all existing neurons - use comprehension to make it quicker
                #
                _ = [self.normaliser.renormalise(graph=self.neural_gas.nodes[neuron_key]['_value'], create_new=False)
                     for neuron_key in self.neural_gas.nodes]

            # calc the distance of the training graph to the existing neurons
            #
            distances = [(neuron_key,
                          self.neural_gas.nodes[neuron_key]['_value'].compare_graph(graph_to_compare=t_graph,
                                                                                    compare_node_types=self.search_node_types),
                          self.neural_gas.nodes[neuron_key]['n_bmu'])
                         for neuron_key in self.neural_gas.nodes]

            # sort in ascending order of actual distance and descending order of number of times bmu
            #
            distances.sort(key=lambda x: (x[1]['graph']['actual'], -x[2]))

            # the bmu is the closest and thus the top of the list
            #
            bmu_key = distances[0][0]
            bmu_distance = distances[0][1]['graph']['actual']

            por['bmu_key'] = bmu_key
            por['bmu_distance'] = bmu_distance
            por['bmu_distance_threshold'] = self.neural_gas.nodes[bmu_key]['threshold']

            # if the distance is larger than the neuron's threshold then add a new neuron
            #
            if bmu_distance > self.neural_gas.nodes[bmu_key]['threshold']:

                # add new neuron
                # distance threshold is mid point between new neuron and bmu
                #
                distance_threshold = bmu_distance / 2.0

                new_neuron_key = self.add_neuron(graph=t_graph, distance_threshold=distance_threshold)

                por['new_neuron_key'] = new_neuron_key

                # connect the new neuron to the bmu neuron and remember the distance
                #
                self.neural_gas.set_edge(source_key=bmu_key, edge_type='NN', target_key=new_neuron_key, distance=bmu_distance)

                # increase the distance threshold of the existing (bmu neuron) if required
                #
                if distance_threshold > self.neural_gas.nodes[bmu_key]['threshold']:
                    self.neural_gas.nodes[bmu_key]['threshold'] = distance_threshold

                if self.delete_old_neurons:
                    # get first neuron that has aged enough to be deleted
                    #
                    neuron_to_deactivate = []
                    for neuron_key in self.neural_gas.nodes:
                        if neuron_key not in [new_neuron_key, bmu_key]:

                            # decay the activation with rate the depends on its current learn_rate and the slow_alpha
                            #
                            self.neural_gas.nodes[neuron_key]['activation'] -= (self.neural_gas.nodes[neuron_key]['activation'] * self.slow_alpha * self.neural_gas.nodes[neuron_key]['learn_rate'])
                            if self.neural_gas.nodes[neuron_key]['activation'] < self.prune_threshold:
                                neuron_to_deactivate.append(neuron_key)

                                # only need first 1 so beak out of loop
                                #
                                break

                    if len(neuron_to_deactivate) > 0:
                        self.neural_gas.remove_node(neuron_to_deactivate[0])
                        por['deleted_neuron_key'] = neuron_to_deactivate[0]

            else:

                # the data is close enough to the bmu to be mapped
                # so update the bmu neuron attributes
                #
                self.neural_gas.nodes[bmu_key]['n_bmu'] += 1
                self.neural_gas.nodes[bmu_key]['last_bmu'] = self.update_id

                # a neuron's error for mapped data is the exponential moving average of the distance.
                #
                if self.neural_gas.nodes[bmu_key]['ema_error'] is None:
                    self.neural_gas.nodes[bmu_key]['ema_error'] = bmu_distance
                else:
                    self.neural_gas.nodes[bmu_key]['ema_error'] += ((bmu_distance - self.neural_gas.nodes[bmu_key]['ema_error']) * self.slow_alpha)

                # reduce the distance threshold towards the error average
                #
                self.neural_gas.nodes[bmu_key]['threshold'] += (self.neural_gas.nodes[bmu_key]['ema_error'] - self.neural_gas.nodes[bmu_key]['threshold']) * self.slow_alpha

                # learn the generalised graph
                #
                self.neural_gas.nodes[bmu_key]['_value'].learn(graph_to_learn=t_graph,
                                                               learn_rate=self.neural_gas.nodes[bmu_key]['learn_rate'],
                                                               learn_node_types=learn_node_types)

                # reset the bmu activation to full strength
                #
                self.neural_gas.nodes[bmu_key]['activation'] = 1.0

                updated_neurons = set()

                updated_neurons.add(bmu_key)

                if len(distances) > 1:

                    nn_idx = 1
                    finished = False
                    while not finished:

                        nn_key = distances[nn_idx][0]
                        nn_distance = distances[nn_idx][1]['graph']['actual']

                        # if the neuron is close enough to the incoming data
                        #
                        if nn_distance < self.neural_gas.nodes[nn_key]['threshold']:

                            updated_neurons.add(nn_key)
                            por['nn_neurons'].append({'nn_distance': nn_distance, 'nn_key': nn_key, 'nn_distance_threshold': self.neural_gas.nodes[nn_key]['threshold']})

                            self.neural_gas.nodes[nn_key]['n_runner_up'] += 1
                            self.neural_gas.nodes[nn_key]['last_runner_up'] = self.update_id

                            # reset the neighbour activation to full strength
                            #
                            self.neural_gas.nodes[nn_key]['activation'] = 1.0

                            # the learning rate for a neighbour needs to be much less that the bmu - hence the product of learning rates and 0.1 factor
                            #
                            nn_learn_rate = self.neural_gas.nodes[bmu_key]['learn_rate'] * self.neural_gas.nodes[nn_key]['learn_rate'] * 0.1

                            # learn the generalised graph
                            #
                            self.neural_gas.nodes[nn_key]['_value'].learn(graph_to_learn=t_graph,
                                                                          learn_rate=nn_learn_rate,
                                                                          learn_node_types=learn_node_types)
                            nn_idx += 1
                            if nn_idx >= len(distances):
                                finished = True
                        else:
                            finished = True

                    # recalculate the distances between updated neurons
                    #
                    edges_to_process = set()
                    for neuron_key in updated_neurons:
                        for edge_key in self.neural_gas.nodes[neuron_key]['_edges']:
                            if edge_key not in edges_to_process:

                                if self.neural_gas.edges[edge_key]['_source'] != neuron_key:
                                    nn_key = self.neural_gas.edges[edge_key]['_source']
                                else:
                                    nn_key = self.neural_gas.edges[edge_key]['_target']

                                distance = self.neural_gas.nodes[neuron_key]['_value'].compare_graph(graph_to_compare=self.neural_gas.nodes[nn_key]['_value'],
                                                                                                     compare_node_types=self.search_node_types)

                                # set the distance in the edge
                                #
                                self.neural_gas.edges[edge_key]['distance'] = distance['graph']['actual']
                                edges_to_process.add(edge_key)

                # decay the learning rate so that this neuron learns more slowly the more it gets mapped too
                #
                self.neural_gas.nodes[bmu_key]['learn_rate'] -= self.neural_gas.nodes[bmu_key]['learn_rate'] * self.slow_alpha

            anomaly, motif = self.update_gas_error(bmu_key=bmu_key, bmu_distance=bmu_distance, ref_id=ref_id)
            por['anomaly'] = anomaly
            por['motif'] = motif

        por['nos_neurons'] = len(self.neural_gas.nodes)

        return por

    def query(self, query_graph, bmu_only: bool = True) -> Por:

        if self.normaliser is not None:

            q_graph, _ = self.normaliser.normalise(graph=query_graph, create_new=True)
        else:
            q_graph = query_graph

        # get the types of nodes to search for
        #
        search_node_types = {q_graph.nodes[node_key]['_type'] for node_key in q_graph.nodes}

        # calc the distance of the training graph to the existing neurons
        #
        distances = [(neuron_key,
                      self.neural_gas.nodes[neuron_key]['_value'].compare_graph(graph_to_compare=q_graph,
                                                                                compare_node_types=search_node_types),
                      self.neural_gas.nodes[neuron_key]['n_bmu'],
                      self.neural_gas.nodes[neuron_key]['_value'],
                      self.neural_gas.nodes[neuron_key]['threshold'])
                     for neuron_key in self.neural_gas.nodes]

        # sort in ascending order of distance and descending order of number of times bmu
        #
        distances.sort(key=lambda x: (x[1]['graph']['actual'], -x[2]))

        # get closest neuron and all other 'activated neurons'
        #
        activated_neurons = [distances[n_idx]
                             for n_idx in range(len(distances))
                             if n_idx == 0 or distances[n_idx][1]['graph']['actual'] <= distances[n_idx][4]]

        por = {'sogam': self.fabric_name,
               'domain': self.domain}

        if len(activated_neurons) > 0:

            sum_distance = sum([n[1]['graph']['actual'] for n in activated_neurons])

            # select the bmu
            #
            if bmu_only or sum_distance == 0 or len(activated_neurons) == 1:
                por['graph'] = activated_neurons[0][3]

                if sum_distance > 0 and len(activated_neurons) > 1:
                    por['neurons'] = [{'neuron_key': n[0], 'weight': 1 - (n[1]['graph']['actual'] / sum_distance)}
                                      for n in activated_neurons]
                else:
                    por['neurons'] = [{'neuron_key': n[0], 'weight': 1.0} for n in activated_neurons]

            # else create a weighted average of neurons
            #
            else:
                por['graph'] = AMHGraph()
                por['neurons'] = []
                for n in activated_neurons:
                    weight = 1 - (n[1]['graph']['actual'] / sum_distance)
                    por['graph'].merge_graph(graph_to_merge=n[3], weight=weight)
                    por['neurons'].append({'neuron_key': n[0], 'weight': weight})

            if self.normaliser is not None:
                por['graph'] = self.normaliser.denormalise(graph=por['graph'])

        return por


if __name__ == '__main__':

    ng = AMGas(fabric_name='test',
               domain='trades',
               anomaly_threshold_factor=4.0,
               fast_alpha=0.7,
               prune_threshold=0.01,
               normaliser=True
               )

    t1 = AMHGraph(directional=True)
    trade_key = t1.set_node(node_type='TRADE', value='*')
    date_key = t1.set_node(node_type='DATE', value='22-11-66')
    platform_key = t1.set_node(node_type='PLATFORM', value='A')
    volume_key = t1.set_node(node_type='VOLUME', value=100)

    t1.set_edge(source_key=trade_key, target_key=platform_key, edge_type='HAS_PLATFORM')
    t1.set_edge(source_key=trade_key, target_key=date_key, edge_type='HAS_DATE')
    t1.set_edge(source_key=trade_key, target_key=volume_key, edge_type='HAS_VOLUME')

    node_types = {t1.nodes[node_key]['_type'] for node_key in t1.nodes}

    p1 = ng.train(training_graph=t1,
                  ref_id='t1',
                  search_node_types=node_types,
                  learn_node_types=node_types)

    t2 = AMHGraph(directional=True)
    trade_key = t2.set_node(node_type='TRADE', value='*')
    date_key = t2.set_node(node_type='DATE', value='22-11-66')
    platform_key = t2.set_node(node_type='PLATFORM', value='B')
    volume_key = t2.set_node(node_type='VOLUME', value=50)

    t2.set_edge(source_key=trade_key, target_key=platform_key, edge_type='HAS_PLATFORM')
    t2.set_edge(source_key=trade_key, target_key=date_key, edge_type='HAS_DATE')
    t2.set_edge(source_key=trade_key, target_key=volume_key, edge_type='HAS_VOLUME')

    p2 = ng.train(training_graph=t2,
                  ref_id='t2',
                  search_node_types=node_types,
                  learn_node_types=node_types)

    t3 = AMHGraph(directional=True)
    trade_key = t3.set_node(node_type='TRADE', value='*')
    date_key = t3.set_node(node_type='DATE', value='22-11-66')
    platform_key = t3.set_node(node_type='PLATFORM', value='A')
    volume_key = t3.set_node(node_type='VOLUME', value=75)

    t3.set_edge(source_key=trade_key, target_key=platform_key, edge_type='HAS_PLATFORM')
    t3.set_edge(source_key=trade_key, target_key=date_key, edge_type='HAS_DATE')
    t3.set_edge(source_key=trade_key, target_key=volume_key, edge_type='HAS_VOLUME')

    p3 = ng.train(training_graph=t3,
                  ref_id='t3',
                  search_node_types=node_types,
                  learn_node_types=node_types)

    t4 = AMHGraph(directional=True)
    trade_key = t4.set_node(node_type='TRADE', value='*')
    date_key = t4.set_node(node_type='DATE', value='22-11-66')
    platform_key = t4.set_node(node_type='PLATFORM', value='B')
    volume_key = t4.set_node(node_type='VOLUME', value=60)

    t4.set_edge(source_key=trade_key, target_key=platform_key, edge_type='HAS_PLATFORM')
    t4.set_edge(source_key=trade_key, target_key=date_key, edge_type='HAS_DATE')
    t4.set_edge(source_key=trade_key, target_key=volume_key, edge_type='HAS_VOLUME')

    p4 = ng.train(training_graph=t4,
                  ref_id='t4',
                  search_node_types=node_types,
                  learn_node_types=node_types)

    t5 = AMHGraph(directional=True)
    trade_key = t5.set_node(node_type='TRADE', value='*')
    date_key = t5.set_node(node_type='DATE', value='22-11-66')
    platform_key = t5.set_node(node_type='PLATFORM', value='A')
    volume_key = t5.set_node(node_type='VOLUME', value=110)

    t5.set_edge(source_key=trade_key, target_key=platform_key, edge_type='HAS_PLATFORM')
    t5.set_edge(source_key=trade_key, target_key=date_key, edge_type='HAS_DATE')
    t5.set_edge(source_key=trade_key, target_key=volume_key, edge_type='HAS_VOLUME')

    p5 = ng.train(training_graph=t5,
                  ref_id='t5',
                  search_node_types=node_types,
                  learn_node_types=node_types)

    j_ng = ng.to_dict(denormalise=True)

    t6 = AMHGraph(directional=True)
    trade_key = t6.set_node(node_type='TRADE', value='*')
    volume_key = t6.set_node(node_type='VOLUME', value=90)
    t6.set_edge(source_key=trade_key, target_key=volume_key, edge_type='HAS_VOLUME')

    q_por_bmu = ng.query(query_graph=t6, bmu_only=True)
    q_por_weav = ng.query(query_graph=t6, bmu_only=False)

    print('finished')

