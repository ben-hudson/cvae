import networkx as nx
import numpy as np
import pyepo
import pyepo.data
import pyepo.model
import torch

from collections import namedtuple
from matplotlib import pyplot as plt

PyEPODatapoint = namedtuple("PyEPODatapoint", "feats costs sols objs cost_params")


class PyEPODataset(pyepo.data.dataset.optDataset):
    def __init__(self, model, feats, costs, cost_params):
        super().__init__(model, feats, costs)
        # super sets self.feats and self.costs
        self.sols = torch.FloatTensor(self.sols)
        self.objs = torch.FloatTensor(self.objs)
        self.cost_params = cost_params

        self.is_integer = ((self.sols == 0) | (self.sols == 1)).all()

        self.means = PyEPODatapoint(
            self.feats.mean(dim=0, keepdim=True),
            self.costs.mean(dim=0, keepdim=True),
            self.sols.mean(dim=0, keepdim=True),
            self.objs.mean(dim=0, keepdim=True),
            self.cost_params.mean(dim=0, keepdim=True),
        )

        self.scales = PyEPODatapoint(
            self.feats.std(dim=0, correction=0, keepdim=True),
            self.costs.std(dim=0, correction=0, keepdim=True),
            self.sols.std(dim=0, correction=0, keepdim=True),
            self.objs.std(dim=0, correction=0, keepdim=True),
            self.cost_params.std(dim=0, correction=0, keepdim=True),
        )

        # if solutions are binary, we don't normalize them
        if self.is_integer:
            self.means.sols[:] = 0.0
            self.scales.sols[:] = 1.0

        # prevent division by 0, just like sklearn.preprocessing.StandardScaler
        for scale in self.scales:
            scale[torch.isclose(scale, torch.zeros_like(scale))] = 1.0

        for field, mean in self.means._asdict().items():
            assert not mean.requires_grad, f"{field} mean requires grad"

        for field, scale in self.scales._asdict().items():
            assert not scale.requires_grad, f"{field} scale requires grad"

    def norm(self, **kwargs):
        means = self.means._asdict()
        scales = self.scales._asdict()
        normed_values = {}

        for name in PyEPODatapoint._fields:
            if name in kwargs:
                value = kwargs[name]
                normed_values[name] = (value - means[name].to(value.device)) / scales[name].to(value.device)
            else:
                normed_values[name] = None

        return PyEPODatapoint(**normed_values)

    def unnorm(self, **kwargs):
        means = self.means._asdict()
        scales = self.scales._asdict()
        unnormed_values = {}

        for name in PyEPODatapoint._fields:
            if name in kwargs:
                value = kwargs[name]
                unnormed_values[name] = means[name].to(value.device) + scales[name].to(value.device) * value
            else:
                unnormed_values[name] = None

        return PyEPODatapoint(**unnormed_values)

    def __getitem__(self, index):
        return PyEPODatapoint(
            self.feats[index], self.costs[index], self.sols[index], self.objs[index], self.cost_params[index]
        )


def render_shortestpath(data_model, sol):
    graph = nx.Graph()
    for (i, j), x in zip(data_model._getArcs(), sol):
        graph.add_edge(i, j, width=x)

    fig = plt.figure(figsize=(1, 1))
    ax = plt.gca()
    ax.set_aspect("equal")

    nodes_x, nodes_y = np.unravel_index(graph.nodes, data_model.grid)
    node_pos = {n: (x, y) for n, x, y in zip(graph.nodes, nodes_x, nodes_y)}
    nx.draw_networkx_nodes(graph, pos=node_pos, node_size=25, node_color="grey")
    # nx.draw_networkx_labels(graph, pos=node_pos, ax=ax)

    edge_widths = nx.get_edge_attributes(graph, "width")
    nx.draw_networkx_edges(graph, pos=node_pos, edgelist=list(edge_widths.keys()), width=list(edge_widths.values()))

    img = fig_to_rgb_tensor(fig)
    plt.close(fig)
    return img


def fig_to_rgb_tensor(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_np = data.reshape((int(h), int(w), -1)).copy()
    # matplotlib puts channels last, pytorch puts channels first
    img_pt = torch.FloatTensor(img_np).permute(2, 0, 1)
    return img_pt
