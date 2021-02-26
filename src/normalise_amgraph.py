#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Tuple
from copy import deepcopy

from sogam.am_graph import AMGraph




class NormaliseAMGraph(object):
    def __init__(self, normalise_amgraph=None):

        if normalise_amgraph is None:
            self._normalise_groups = {}
            self._min_max = {}

        elif isinstance(normalise_amgraph, NormaliseAMGraph):
            self._normalise_groups = deepcopy(normalise_amgraph._normalise_groups)
            self._min_max = deepcopy(normalise_amgraph._min_max)

        else:
            self._normalise_groups = normalise_amgraph['normalise_groups']
            self._min_max = normalise_amgraph['min_max']

    def to_dict(self) -> dict:
        n_dict = {'min_max': deepcopy(self._min_max),
                  'normalise_groups': deepcopy(self._normalise_groups),
                  }
        return n_dict

    def normalise(self, graph: AMGraph, create_new: bool = True) -> Tuple[AMGraph, bool]:

        if create_new:
            # create a new graph that will hold the normalised values
            #
            n_graph = AMGraph(graph=graph)
        else:
            n_graph = graph

        new_min_max = False
        if not n_graph.normalised:

            # in this loop determine the min and max of each edge types normalisation group
            #
            for triple_key in n_graph.edges:

                numeric = n_graph.edges[triple_key]['_numeric']

                if numeric is not None:

                    # if this edge type isn't in the normalise groups dict then add
                    #
                    if n_graph.edges[triple_key]['_type'] not in self._normalise_groups:
                        self._normalise_groups[n_graph.edges[triple_key]['_type']] = n_graph.edges[triple_key]['_type']

                    group = self._normalise_groups[n_graph.edges[triple_key]['_type']]

                    # update the global min and max for this triple
                    #
                    if group not in self._min_max:
                        self._min_max[group] = {'min': numeric - 0.001, 'max': numeric, 'prev_min': numeric - 0.001, 'prev_max': numeric}
                        new_min_max = True

                    elif numeric < self._min_max[group]['min']:
                        self._min_max[group]['prev_min'] = self._min_max[group]['min']
                        self._min_max[group]['prev_max'] = self._min_max[group]['max']
                        self._min_max[group]['min'] = numeric
                        new_min_max = True

                    elif numeric > self._min_max[group]['max']:
                        self._min_max[group]['prev_min'] = self._min_max[group]['min']
                        self._min_max[group]['prev_max'] = self._min_max[group]['max']
                        self._min_max[group]['max'] = numeric
                        new_min_max = True

            # in this loop normalise the graph knowing the min and max of each edge type normalisation group
            #
            for triple_key in n_graph.edges:

                numeric = n_graph.edges[triple_key]['_numeric']

                if numeric is not None:

                    n_graph.normalised = True

                    group = self._normalise_groups[n_graph.edges[triple_key]['_type']]

                    norm_numeric = ((numeric - self._min_max[group]['min']) /
                                    (self._min_max[group]['max'] - self._min_max[group]['min']))

                    n_graph.edges[triple_key]['_numeric'] = norm_numeric

        return n_graph, new_min_max

    def denormalise(self, graph: AMGraph, create_new: bool = True, prev_min_max=False):

        if create_new:
            # create a new graph that will hold the denormalised values
            #
            dn_graph = AMGraph(graph=graph)
        else:
            dn_graph = graph

        if dn_graph.normalised:

            dn_graph.normalised = False
            for triple_key in dn_graph.edges:

                numeric = dn_graph.edges[triple_key]['_numeric']

                if numeric is not None:
                    # get the group for triple's edge type
                    #
                    group = self._normalise_groups[dn_graph.edges[triple_key]['_type']]

                    if prev_min_max:
                        # denormalise first using previous values
                        #
                        dn_numeric = (numeric * (self._min_max[group]['prev_max'] - self._min_max[group]['prev_min']) + self._min_max[group]['prev_min'])
                    else:
                        # denormalise first using current values
                        #
                        dn_numeric = (numeric * (self._min_max[group]['max'] - self._min_max[group]['min']) + self._min_max[group]['min'])

                    dn_graph.edges[triple_key]['_numeric'] = dn_numeric

        return dn_graph

    def renormalise(self, graph: AMGraph, create_new: bool = True) -> AMGraph:

        dn_graph = self.denormalise(graph=graph, create_new=create_new, prev_min_max=True)

        n_graph, _ = self.normalise(graph=dn_graph, create_new=create_new)

        return n_graph


if __name__ == '__main__':
    pass
