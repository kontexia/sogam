#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from sogam.am_graph import AMGraph
from sogam.am_gas import AMGas
from sogam.am_gas_viz import plot_gas, plot_pors
from sklearn.datasets import make_moons, make_swiss_roll
import json
from sogam.am_fabric import AMFabric


def moon_test():
    data_set, labels = make_moons(n_samples=200,
                                  noise=0.05,
                                  random_state=0)

    training_graphs = []

    for idx in range(len(data_set)):
        t_g = AMGraph()

        t_g.set_edge(triple=(('space', '*'), ('has_datum', None, None), ('datum', 'x')),
                     numeric=data_set[idx][0])
        t_g.set_edge(triple=(('space', '*'), ('has_datum', None, None), ('datum', 'y')),
                     numeric=data_set[idx][1])
        t_g.set_edge(triple=(('space', '*'), ('has_label', None, None), ('label', str(labels[idx]))))

        training_graphs.append(t_g)

    ng = AMGas(fabric_name='MoonTest',
               domain='Moon',
               anomaly_threshold_factor=4.0,
               fast_alpha=0.9,

               prune_threshold=0.01,
               audit=False,
               normalise=True)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = ng.train(training_graph=training_graphs[t_idx],
                       ref_id=str(t_idx),
                       search_edge_types={'has_datum', 'has_label'},
                       learn_edge_types={'has_datum', 'has_label'})
        pors.append(por)

    ng_dict = ng.to_dict(denormalise=True)

    plot_gas(gas=ng_dict,
             raw_data=data_set,
             xyz_triples=[(('space', '*'), ('has_datum', None, None), ('datum', 'x')),
                          (('space', '*'), ('has_datum', None, None), ('datum', 'y'))],
             colour_edgeType='has_label')

    plot_pors(pors=pors)

    print('Finished')


def swiss_roll_test():

    data_set, labels = make_swiss_roll(n_samples=400, noise=0.05, random_state=0, )

    training_graphs = []

    for idx in range(len(data_set)):
        t_g = AMGraph()

        t_g.set_edge(triple=(('space', '*'), ('has_datum', None, None), ('datum', 'x')),
                     numeric=data_set[idx][0])
        t_g.set_edge(triple=(('space', '*'), ('has_datum', None, None), ('datum', 'y')),
                     numeric=data_set[idx][1])

        t_g.set_edge(triple=(('space', '*'), ('has_datum', None, None), ('datum', 'z')),
                     numeric=data_set[idx][2])

        training_graphs.append(t_g)

    ng = AMGas(fabric_name='SwissRollTest',
               domain='Swiss',
               anomaly_threshold_factor=4.0,
               fast_alpha=0.9,
               prune_threshold=0.01,
               audit=False,
               normalise=True)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = ng.train(training_graph=training_graphs[t_idx],
                       ref_id=str(t_idx),
                       search_edge_types={'has_datum', 'has_label'},
                       learn_edge_types={'has_datum', 'has_label'})
        pors.append(por)

    ng_dict = ng.to_dict(denormalise=True)

    plot_gas(gas=ng_dict,
             raw_data=data_set,
             xyz_triples=[(('space', '*'), ('has_datum', None, None), ('datum', 'x')),
                          (('space', '*'), ('has_datum', None, None), ('datum', 'y')),
                          (('space', '*'), ('has_datum', None, None), ('datum', 'z'))],
             colour_edgeType=None)

    plot_pors(pors=pors)

    print('Finished')


def square_test():
    training_graphs = []

    for cycles in range(8):
        for low_value in range(3):
            t_g = AMGraph()

            t_g.set_edge(triple=(('space', '*'), ('has_datum', None, None), ('datum', 'x')),
                         numeric=0.0)
            t_g.set_edge(triple=(('space', '*'), ('has_label', None, None), ('label', 'Low')))

            training_graphs.append(t_g)

        for high_value in range(5):
            t_g = AMGraph()

            t_g.set_edge(triple=(('space', '*'), ('has_datum', None, None), ('datum', 'x')),
                         numeric=10.0)
            t_g.set_edge(triple=(('space', '*'), ('has_label', None, None), ('label', 'High')))

            training_graphs.append(t_g)

    ng = AMGas(fabric_name='SquareTest',
               domain='Square',
               anomaly_threshold_factor=4.0,
               fast_alpha=0.9,

               prune_threshold=0.01,
               audit=False,
               normalise=True)

    pors = []

    for t_idx in range(len(training_graphs)):
        por = ng.train(training_graph=training_graphs[t_idx],
                       ref_id=str(t_idx),
                       search_edge_types={'has_datum', 'has_label'},
                       learn_edge_types={'has_datum', 'has_label'})
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
        t_g = AMGraph()

        r_data = []
        for field in ['r', 'g', 'b']:
            t_g.set_edge(triple=(('trade', '*'), ('has_rgb', None, None), ('rgb', field)),
                         numeric=record[field])
            t_g.set_edge(triple=(('trade', '*'), ('has_colour', None, None), ('colour', record['label'])))
            r_data.append(record[field])

        training_graphs[record['client']].append((record['trade_id'], t_g, r_data))

    n_cycles = 10

    gases = {}
    for client in training_graphs:
        pors = []
        ng = AMGas(fabric_name='Colours',
                   domain=client,
                   anomaly_threshold_factor=4.0,
                   fast_alpha=0.5,
                   prune_threshold=0.01,
                   audit=False,
                   normalise=True)
        gases[client] = {'ng': ng}

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
        t_g = AMGraph()

        r_data = []
        for field in ['r', 'g', 'b']:
            t_g.set_edge(triple=(('trade', '*'), ('has_rgb', None, None), ('rgb', field)),
                         numeric=record[field])
            t_g.set_edge(triple=(('trade', '*'), ('has_colour', None, None), ('colour', record['label'])))
            r_data.append(record[field])

        training_graphs[record['client']].append((record['trade_id'], t_g, r_data))

    n_cycles = 10
    gases = {}
    for client in training_graphs:
        pors = []
        ng = AMFabric(fabric_name=client,
                      anomaly_threshold_factor=3.0,
                      fast_alpha=0.5,
                      prune_threshold=0.01,
                      audit=False,
                      normalise=True)
        gases[client] = {'ng': ng}

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


if __name__ == '__main__':
    colours_fabric_test()

    colours_test()
    moon_test()
    swiss_roll_test()
    square_test()
