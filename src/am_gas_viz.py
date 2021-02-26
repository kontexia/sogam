#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from src.am_graph import get_triple_key

import plotly.graph_objects as go


def plot_gas(gas, raw_data, xyz_triples, colour_edgeType):

    node_x = []
    node_y = []
    node_z = []

    next_color = 10
    colour_labels = {}
    node_colour = []
    node_label = []
    node_size = []

    edge_x = []
    edge_y = []
    edge_z = []

    x_min_max = {'min': None, 'max': None}
    y_min_max = {'min': None, 'max': None}
    z_min_max = {'min': None, 'max': None}

    xyz_triples = [get_triple_key(triple_id=triple_id, directional=True) for triple_id in xyz_triples]

    for neuron_key in gas['neural_gas']['nodes']:
        node_x.append(gas['neural_gas']['nodes'][neuron_key]['generalised_graph']['edges'][xyz_triples[0]]['_numeric'])
        node_y.append(gas['neural_gas']['nodes'][neuron_key]['generalised_graph']['edges'][xyz_triples[1]]['_numeric'])
        if len(xyz_triples) == 3:
            node_z.append(gas['neural_gas']['nodes'][neuron_key]['generalised_graph']['edges'][xyz_triples[2]]['_numeric'])
        else:
            node_z.append(0.0)

        if colour_edgeType is not None:

            colour_triple = [gas['neural_gas']['nodes'][neuron_key]['generalised_graph']['edges'][triple_key]['_target']
                             for triple_key in gas['neural_gas']['nodes'][neuron_key]['generalised_graph']['edges']
                             if gas['neural_gas']['nodes'][neuron_key]['generalised_graph']['edges'][triple_key]['_type'] == colour_edgeType]

            if colour_triple[0] not in colour_labels:
                colour = next_color
                colour_labels[colour_triple[0]] = next_color
                next_color += 1
            else:
                colour = colour_labels[colour_triple[0]]
            node_colour.append(colour)
        else:
            if x_min_max['max'] is None or node_x[-1] > x_min_max['max']:
                x_min_max['max'] = node_x[-1]
            if x_min_max['min'] is None or node_x[-1] < x_min_max['min']:
                x_min_max['min'] = node_x[-1]
            if y_min_max['max'] is None or node_y[-1] > y_min_max['max']:
                y_min_max['max'] = node_y[-1]
            if y_min_max['min'] is None or node_y[-1] < y_min_max['min']:
                y_min_max['min'] = node_y[-1]
            if z_min_max['max'] is None or node_z[-1] > z_min_max['max']:
                z_min_max['max'] = node_z[-1]
            if z_min_max['min'] is None or node_z[-1] < z_min_max['min']:
                z_min_max['min'] = node_z[-1]

        node_label.append(neuron_key)
        node_size.append(10 + gas['neural_gas']['nodes'][neuron_key]['n_bmu'])

    if colour_edgeType is None:
        for idx in range(len(node_x)):
            r = max(min(int(255 * ((node_x[idx] - x_min_max['min']) / (x_min_max['max'] - x_min_max['min']))), 255), 0)
            g = max(min(int(255 * ((node_y[idx] - y_min_max['min']) / (y_min_max['max'] - y_min_max['min']))), 255), 0)
            b = max(min(int(255 * ((node_z[idx] - z_min_max['min']) / (z_min_max['max'] - z_min_max['min']))), 255), 0)

            node_colour.append(f'rgb({r},{g},{b})')

    for triple_key in gas['neural_gas']['edges']:

        edge_x.append(gas['neural_gas']['nodes'][gas['neural_gas']['edges'][triple_key]['_source']]['generalised_graph']['edges'][xyz_triples[0]]['_numeric'])
        edge_x.append(gas['neural_gas']['nodes'][gas['neural_gas']['edges'][triple_key]['_target']]['generalised_graph']['edges'][xyz_triples[0]]['_numeric'])
        edge_x.append(None)

        edge_y.append(gas['neural_gas']['nodes'][gas['neural_gas']['edges'][triple_key]['_source']]['generalised_graph']['edges'][xyz_triples[1]]['_numeric'])
        edge_y.append(gas['neural_gas']['nodes'][gas['neural_gas']['edges'][triple_key]['_target']]['generalised_graph']['edges'][xyz_triples[1]]['_numeric'])
        edge_y.append(None)

        if len(xyz_triples) == 3:
            edge_z.append(gas['neural_gas']['nodes'][gas['neural_gas']['edges'][triple_key]['_source']]['generalised_graph']['edges'][xyz_triples[2]]['_numeric'])
            edge_z.append(gas['neural_gas']['nodes'][gas['neural_gas']['edges'][triple_key]['_target']]['generalised_graph']['edges'][xyz_triples[2]]['_numeric'])

        else:
            edge_z.append(0.0)
            edge_z.append(0.0)

        edge_z.append(None)

    raw_x = []
    raw_y = []
    raw_z = []
    raw_size = []
    raw_colour = []
    for idx in range(len(raw_data)):
        raw_x.append(raw_data[idx][0])
        raw_y.append(raw_data[idx][1])
        if len(raw_data[idx]) == 3:
            raw_z.append(raw_data[idx][2])
        else:
            raw_z.append(0.0)
        raw_size.append(5)
        raw_colour.append(1)

    raw_scatter = go.Scatter3d(x=raw_x, y=raw_y, z=raw_z, mode='markers',  marker=dict(size=3, color=raw_colour, opacity=1.0, symbol='square'))
    neuron_scatter = go.Scatter3d(x=node_x, y=node_y, z=node_z, hovertext=node_label, mode='markers+text', marker=dict(size=node_size, color=node_colour, opacity=0.7))
    edge_scatter = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(width=1, color='grey'))

    fig = go.Figure(data=[raw_scatter, edge_scatter, neuron_scatter])

    fig.update_layout(width=1200, height=1200,
                      title=dict(text=f'{gas["fabric_name"]} {gas["domain"]} Nos Neurons: {len(gas["neural_gas"]["nodes"])}'))
    fig.show()


def plot_pors(pors):
    anomaly_threshold = []
    motif_threshold = []
    anomalies = []
    motifs = []
    ema_error = []
    ema_stdev = []

    x_values = []
    error = []
    nos_neurons = []

    for x in range(len(pors)):
        x_values.append(x)
        error.append(pors[x]['bmu_distance'])
        nos_neurons.append(pors[x]['nos_neurons'] / pors[-1]['nos_neurons'])
        anomaly_threshold.append(pors[x]['anomaly_threshold'])
        motif_threshold.append(pors[x]['motif_threshold'])
        ema_error.append(pors[x]['ema_error'])
        ema_stdev.append(pow(pors[x]['ema_variance'], 0.5))

        if pors[x]['anomaly']:
            anomalies.append(pors[x]['bmu_distance'])
        else:
            anomalies.append(None)
        if pors[x]['motif']:
            motifs.append(pors[x]['bmu_distance'])
        else:
            motifs.append(None)

    error_scatter = go.Scatter(x=x_values, y=error, mode='lines', line=dict(width=2, color='black'))
    nos_neurons_scatter = go.Scatter(x=x_values, y=nos_neurons, mode='lines', line=dict(width=2, color='orange'))

    anomaly_threshold_scatter = go.Scatter(x=x_values, y=anomaly_threshold, mode='lines', line=dict(width=2, color='red'))
    motif_threshold_scatter = go.Scatter(x=x_values, y=motif_threshold, mode='lines', line=dict(width=2, color='green'))
    ema_error_scatter = go.Scatter(x=x_values, y=ema_error, mode='lines', line=dict(width=2, color='blue'))
    ema_stdev_scatter = go.Scatter(x=x_values, y=ema_stdev, mode='lines', line=dict(width=2, color='purple'))

    anomalies_scatter = go.Scatter(x=x_values, y=anomalies, mode='markers', marker=dict(size=10, color='red', opacity=1.0, symbol='square'))
    motifs_scatter = go.Scatter(x=x_values, y=motifs, mode='markers', marker=dict(size=10, color='green', opacity=1.0, symbol='square'))

    fig = go.Figure(data=[nos_neurons_scatter, error_scatter, anomaly_threshold_scatter, motif_threshold_scatter, ema_error_scatter,ema_stdev_scatter, anomalies_scatter, motifs_scatter])

    fig.update_layout(width=1200, height=1200,
                      title=dict(text=f'{pors[0]["fabric"]} {pors[0]["domain"]} nos neurons: {pors[-1]["nos_neurons"]}'))
    fig.show()


