import networkx as nx
import numpy as np
import pyepo
import torch

from collections import namedtuple
from matplotlib import pyplot as plt
from pyepo.data.dataset import optDataset
from sklearn.utils import compute_class_weight
from typing import Tuple

PyEPOData = namedtuple("PyEPOData", "feats costs sols objs cost_params")


class PyEPODataset(optDataset):
    def __init__(self, model, feats, costs, cost_params):
        super().__init__(model, feats, costs)
        # super sets self.feats and self.costs
        self.sols = torch.FloatTensor(self.sols)
        self.objs = torch.FloatTensor(self.objs)
        self.cost_params = cost_params

        self.is_integer = ((self.sols == 0) | (self.sols == 1)).all()

        self.means = PyEPOData(
            self.feats.mean(dim=0, keepdim=True),
            self.costs.mean(dim=0, keepdim=True),
            self.sols.mean(dim=0, keepdim=True),
            self.objs.mean(dim=0, keepdim=True),
            self.cost_params.mean(dim=0, keepdim=True),
        )

        self.scales = PyEPOData(
            self.feats.std(dim=0, correction=0, keepdim=True),
            self.costs.std(dim=0, correction=0, keepdim=True),
            self.sols.std(dim=0, correction=0, keepdim=True),
            self.objs.std(dim=0, correction=0, keepdim=True),
            self.cost_params.std(dim=0, correction=0, keepdim=True),
        )

        # if solutions are binary, we don't normalize them
        # we do calculate class weights (there are always more 0s than 1s)
        if self.is_integer:
            self.means.sols[:] = 0.0
            self.scales.sols[:] = 1.0
            self.class_weights = compute_class_weight(
                class_weight="balanced", classes=np.array([0, 1]), y=self.sols.flatten().numpy()
            )

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

        for name in PyEPOData._fields:
            if name in kwargs:
                value = kwargs[name]
                normed_values[name] = (value - means[name].to(value.device)) / scales[name].to(value.device)
            else:
                normed_values[name] = None

        return PyEPOData(**normed_values)

    def unnorm(self, **kwargs):
        means = self.means._asdict()
        scales = self.scales._asdict()
        unnormed_values = {}

        for name in PyEPOData._fields:
            if name in kwargs:
                value = kwargs[name]
                unnormed_values[name] = means[name].to(value.device) + scales[name].to(value.device) * value
            else:
                unnormed_values[name] = None

        return PyEPOData(**unnormed_values)

    def __getitem__(self, index):
        return PyEPOData(
            self.feats[index], self.costs[index], self.sols[index], self.objs[index], self.cost_params[index]
        )


def gen_shortestpath_data(
    n_samples: int, n_features: int, grid: Tuple[int], degree: int, noise_width: float, seed: int
):
    feats, expected_costs = pyepo.data.shortestpath.genData(
        n_samples, n_features, grid, deg=degree, noise_width=0, seed=seed
    )

    feats = torch.FloatTensor(feats)
    expected_costs = torch.FloatTensor(expected_costs)

    cost_dist_scales = noise_width / expected_costs.abs()

    cost_samples = torch.distributions.Normal(expected_costs, cost_dist_scales).sample()

    # we apply a samplewise unit norm to the costs and cost distributions
    # this does not affect the solutions, but it does change some metrics
    cost_sample_norms = cost_samples.norm(dim=-1).unsqueeze(-1)
    cost_samples_normed = cost_samples / cost_sample_norms
    # this is a linear transformation, and so it is valid to apply it to the cost distributions
    # see https://en.wikipedia.org/wiki/Normal_distribution#Operations_and_functions_of_normal_variables
    cost_dist_norms = expected_costs.norm(dim=-1).unsqueeze(-1)
    cost_dist_params_normed = torch.cat([expected_costs / cost_dist_norms, cost_dist_scales / cost_dist_norms], dim=-1)

    return feats, cost_samples_normed, cost_dist_params_normed


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
