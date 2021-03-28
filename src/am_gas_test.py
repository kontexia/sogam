#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from src.am_hgraph import AMHGraph, GraphNormaliser
from src.am_gas import AMGas
from src.am_gas_viz import plot_gas, plot_pors
from sklearn.datasets import make_moons, make_swiss_roll
import json
from src.am_fabric import AMFabric


def moon_test():
    data_set, labels = make_moons(n_samples=200,
                                  noise=0.05,
                                  random_state=0)

    training_graphs = []

    norm = {'_types': {'y': 'y', 'x': 'x'},
            '_groups': {'y': {'min': 0, 'max': 0, 'prev_min': 0, 'prev_max': 0},
                        'x': {'min': 0, 'max': 0, 'prev_min': 0, 'prev_max': 0}}}
    for idx in range(len(data_set)):
        t_g = AMHGraph()
        t_g.set_node(node_type='x', value=data_set[idx][0])
        t_g.set_node(node_type='y', value=data_set[idx][1])
        t_g.set_node(node_type='label', value=str(labels[idx]))
        if data_set[idx][0] > norm['_groups']['x']['max']:
            norm['_groups']['x']['max'] = data_set[idx][0]
        elif data_set[idx][0] < norm['_groups']['x']['min']:
            norm['_groups']['x']['min'] = data_set[idx][0]

        if data_set[idx][1] > norm['_groups']['y']['max']:
            norm['_groups']['y']['max'] = data_set[idx][0]
        elif data_set[idx][1] < norm['_groups']['y']['min']:
            norm['_groups']['y']['min'] = data_set[idx][0]

        training_graphs.append(t_g)

    normaliser = GraphNormaliser(normaliser=norm)

    ng = AMGas(fabric_name='MoonTest',
               domain='Moon',
               anomaly_threshold_factor=4.0,
               fast_alpha=0.7,
               prune_threshold=0.001,
               normaliser=normaliser)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = ng.train(training_graph=training_graphs[t_idx],
                       ref_id=str(t_idx),
                       search_node_types={'x', 'y', 'label'},
                       learn_node_types={'x', 'y', 'label'})
        pors.append(por)

    ng_dict = ng.to_dict(denormalise=True)

    plot_gas(gas=ng_dict,
             raw_data=data_set,
             xyz_nodes=['x', 'y'],
             colour_nodes='label')

    plot_pors(pors=pors)

    print('Finished')


def swiss_roll_test():

    data_set, labels = make_swiss_roll(n_samples=400, noise=0.05, random_state=0, )

    training_graphs = []

    norm = {'_types': {'y': 'y', 'x': 'x'},
            '_groups': {'y': {'min': 0, 'max': 10, 'prev_min': 0, 'prev_max': 10},
                        'x': {'min': 0, 'max': 10, 'prev_min': 0, 'prev_max': 10}}}
    for idx in range(len(data_set)):
        t_g = AMHGraph()
        t_g.set_node(node_type='x', value=data_set[idx][0])
        t_g.set_node(node_type='y', value=data_set[idx][1])
        t_g.set_node(node_type='label', value=str(labels[idx]))
        if data_set[idx][0] > norm['_groups']['x']['max']:
            norm['_groups']['x']['max'] = data_set[idx][0]
        elif data_set[idx][0] < norm['_groups']['x']['min']:
            norm['_groups']['x']['min'] = data_set[idx][0]

        if data_set[idx][1] > norm['_groups']['y']['max']:
            norm['_groups']['y']['max'] = data_set[idx][0]
        elif data_set[idx][1] < norm['_groups']['y']['min']:
            norm['_groups']['y']['min'] = data_set[idx][0]

        training_graphs.append(t_g)

    normaliser = GraphNormaliser(normaliser=norm)

    ng = AMGas(fabric_name='SwissRollTest',
               domain='Swiss',
               anomaly_threshold_factor=4.0,
               fast_alpha=0.95,
               prune_threshold=0.001,
               normaliser=normaliser)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = ng.train(training_graph=training_graphs[t_idx],
                       ref_id=str(t_idx),
                       search_node_types={'x', 'y', 'label'},
                       learn_node_types={'x', 'y', 'label'})
        pors.append(por)

    ng_dict = ng.to_dict(denormalise=True)

    plot_gas(gas=ng_dict,
             raw_data=data_set,
             xyz_nodes=['x', 'y'],
             colour_nodes='label')

    plot_pors(pors=pors)

    print('Finished')


def square_test():
    training_graphs = []

    for cycles in range(8):
        for low_value in range(3):
            t_g = AMHGraph()
            t_g.set_node(node_type='y', value=0.0)
            t_g.set_node(node_type='label', value='Low')

            training_graphs.append(t_g)

        for high_value in range(5):
            t_g = AMHGraph()

            t_g.set_node(node_type='y', value=10.0)
            t_g.set_node(node_type='label', value='High')

            training_graphs.append(t_g)

    normaliser = GraphNormaliser(normaliser={'_types': {'y': 'y'},
                                             '_groups': {'y': {'min': 0, 'max': 10, 'prev_min': 0, 'prev_max': 10}}})

    ng = AMGas(fabric_name='SquareTest',
               domain='Square',
               anomaly_threshold_factor=4.0,
               fast_alpha=0.7,

               prune_threshold=0.01,
               normaliser=normaliser)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = ng.train(training_graph=training_graphs[t_idx],
                       ref_id=str(t_idx),
                       search_node_types={'y', 'label'},
                       learn_node_types={'y', 'label'})
        pors.append(por)

    ng_dict = ng.to_dict(denormalise=True)

    plot_pors(pors=pors)

    print('Finished')


def colours_test():
    colours = {'RED': {'r': 255, 'b': 0, 'g': 0},
               'ORANGE': {'r': 255, 'b': 129, 'g': 0},
               'YELLOW': {'r': 255, 'b': 233, 'g': 0},
               'GREEN': {'r': 0, 'b': 202, 'g': 14},
               'BLUE': {'r': 22, 'b': 93, 'g': 239},
               'PURPLE': {'r': 166, 'b': 1, 'g': 214},
               'BROWN': {'r': 151, 'b': 76, 'g': 2},
               'GREY': {'r': 128, 'b': 128, 'g': 128},
               'BLACK': {'r': 0, 'b': 0, 'g': 0},
               'TURQUOISE': {'r': 150, 'b': 255, 'g': 255},
               }

    file_name = '../data/rainbow_trades.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    print('raw data record:', raw_data[:1])

    training_graphs = {}
    for record in raw_data:
        if record['client'] not in training_graphs:
            training_graphs[record['client']] = []
        t_g = AMHGraph()

        r_data = []
        for field in ['r', 'g', 'b']:
            t_g.set_node(node_type=field, value=record[field])
            r_data.append(record[field])

        t_g.set_node(node_type='colour', value=record['label'])

        training_graphs[record['client']].append((record['trade_id'], t_g, r_data))

    n_cycles = 1

    normaliser = GraphNormaliser(normaliser={'_types': {'r': 'r', 'g': 'g', 'b': 'b'},
                                             '_groups': {'r': {'min': 0, 'max': 255, 'prev_min': 0, 'prev_max': 255},
                                                         'g': {'min': 0, 'max': 255, 'prev_min': 0, 'prev_max': 255},
                                                         'b': {'min': 0, 'max': 255, 'prev_min': 0, 'prev_max': 255}}})
    gases = {}
    for client in training_graphs:
        pors = []
        ng = AMGas(fabric_name='Colours',
                   domain=client,
                   anomaly_threshold_factor=6.0,
                   fast_alpha=0.7,
                   prune_threshold=0.00001,
                   normaliser=normaliser)
        gases[client] = {'ng': ng}

        for cycle in range(n_cycles):
            for t_idx in range(len(training_graphs[client])):
                por = ng.train(training_graph=training_graphs[client][t_idx][1],
                               ref_id=str(t_idx),
                               search_node_types={'r', 'g', 'b', 'colour'},
                               learn_node_types={'r', 'g', 'b', 'colour'})
                pors.append(por)

        gases[client]['por'] = pors

        #ng.calc_communities()

        plot_pors(pors=pors)

        ng_dict = ng.to_dict(denormalise=True)

        rn_data = [t_data[2] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)
        plot_gas(gas=ng_dict,
                 raw_data=cycle_data,
                 xyz_nodes=['r', 'g', 'b'],
                 colour_nodes=None)

        print(f'finished {client}')



def colours_fabric_test():
    colours = {'RED': {'r': 255, 'b': 0, 'g': 0},
               'ORANGE': {'r': 255, 'b': 129, 'g': 0},
               'YELLOW': {'r': 255, 'b': 233, 'g': 0},
               'GREEN': {'r': 0, 'b': 202, 'g': 14},
               'BLUE': {'r': 22, 'b': 93, 'g': 239},
               'PURPLE': {'r': 166, 'b': 1, 'g': 214},
               'BROWN': {'r': 151, 'b': 76, 'g': 2},
               'GREY': {'r': 128, 'b': 128, 'g': 128},
               'BLACK': {'r': 0, 'b': 0, 'g': 0},
               'TURQUOISE': {'r': 150, 'b': 255, 'g': 255},
               }

    file_name = '../data/rainbow_trades.json'
    with open(file_name, 'r') as fp:
        raw_data = json.load(fp)

    print('raw data record:', raw_data[:1])

    training_graphs = {}
    for record in raw_data:
        if record['client'] not in training_graphs:
            training_graphs[record['client']] = []
        t_g = AMHGraph()

        r_data = []
        for field in ['r', 'g', 'b']:
            t_g.set_node(node_type=field, value=record[field])
            r_data.append(record[field])

        t_g.set_node(node_type='colour', value=record['label'])

        training_graphs[record['client']].append((record['trade_id'], t_g, r_data))

    n_cycles = 1
    gases = {}
    for client in training_graphs:
        pors = []
        ng = AMFabric(fabric_name=client,
                      anomaly_threshold_factor=6.0,
                      fast_alpha=0.7,
                      prune_threshold=0.0001,
                      audit=False,
                      normalise=True)
        gases[client] = {'ng': ng}

        # TODO
        for cycle in range(n_cycles):
            for t_idx in range(len(training_graphs[client])):
                por = ng.train(training_graph=training_graphs[client][t_idx][1],
                               ref_id=str(t_idx),
                               search_edge_types={'has_rgb', 'has_colour'},
                               learn_edge_types={'has_rgb', 'has_colour'})
                pors.append(por)

        gases[client]['por'] = pors

        ng.calc_communities()

        plot_pors(pors=pors)

        ng_dict = ng.to_dict(denormalise=True)

        rn_data = [t_data[2] for t_data in training_graphs[client]]
        cycle_data = []
        for cycle in range(n_cycles):
            cycle_data.extend(rn_data)
        plot_gas(gas=ng_dict,
                 raw_data=cycle_data,
                 xyz_triples=[(('trade', '*'), ('has_rgb', None, None), ('rgb', 'r')),
                              (('trade', '*'), ('has_rgb', None, None), ('rgb', 'g')),
                              (('trade', '*'), ('has_rgb', None, None), ('rgb', 'b')),
                              ],
                 colour_edgeType=None)
        print(f'finished {client}')


def number_experiments():
    from random import sample
    numbers = [n for n in sample(range(1000), k=60)]

    #numbers = [n for n in range(60)]

    training_graphs = []

    for n in numbers:
        tg = AMHGraph()
        tg.set_node(node_type='quantity', value=n)

        training_graphs.append(tg)

    pors = []
    ng = AMGas(fabric_name='Numbers',
               domain='ints',
               anomaly_threshold_factor=6.0,
               fast_alpha=0.7,
               prune_threshold=0.00001,
               audit=False,
               normalise=True,
               delete_old_neurons=False)

    cycles = 1
    for cycle in range(cycles):
        for t_idx in range(len(training_graphs)):
            por = ng.train(training_graph=training_graphs[t_idx],
                           ref_id=str(t_idx),
                           search_node_types={'quantity'},
                           learn_node_types={'quantity'})
            pors.append(por)

    print('finished')



def category_experiments():

    years = [n for n in range(1960, 2020)]

    training_graphs = []
    for y_idx in range(len(years)):
        tg = AMHGraph()
        source_key = tg.set_node(node_type='year', value=str(years[y_idx]))
        if y_idx > 0 is not None:
            target_key = tg.set_node(node_type='year', value=str(years[y_idx-1]))
            tg.set_edge(source_key=source_key, target_key=target_key, edge_type='prev_year')
            tg.set_edge(source_key=target_key, target_key=source_key, edge_type='next_year')

        training_graphs.append(tg)

    pors = []
    ng = AMGas(fabric_name='years',
               domain='years',
               anomaly_threshold_factor=6.0,
               fast_alpha=0.7,
               prune_threshold=0.00001,
               audit=False,
               normalise=True,
               delete_old_neurons=False)

    cycles = 1
    for cycle in range(cycles):
        for t_idx in range(len(training_graphs)):
            por = ng.train(training_graph=training_graphs[t_idx],
                           ref_id=str(t_idx),
                           search_node_types={'year'},
                           learn_node_types={'year'})
            pors.append(por)

    print('finished')



if __name__ == '__main__':

    #category_experiments()

    #number_experiments()

    #colours_fabric_test()

    colours_test()
    moon_test()
    swiss_roll_test()
    square_test()
