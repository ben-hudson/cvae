import networkx as nx
import torch

from distributions.twopoint import TwoPoint


def get_toy_graph():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 1)
    return G


# the data-generating process is very simple
# the graph has two edges
# one edge cost can take a low value (5) with high probability (0.8) or a high value (20) with low probability (0.2)
# the other edge cost takes an intermediate value (10) with probability 1.
# these two costs switch depending on the binary input feature
def gen_toy_data(n_samples):
    feats = torch.randint(0, 2, size=(n_samples,)).bool()
    cost_dist_params = torch.empty((n_samples, 3, 2))

    rows = torch.arange(n_samples)
    cost_dist_params[rows, :, feats.int()] = torch.FloatTensor([5, 20, 0.2])
    cost_dist_params[rows, :, (~feats).int()] = torch.FloatTensor([0, 10, 1.0])

    cost_dist_params = cost_dist_params.flatten(1)  # this is the form expected by
    cost_lows, cost_highs, cost_probs = cost_dist_params.chunk(3, dim=-1)
    cost_dists = TwoPoint(cost_lows, cost_highs, cost_probs)
    cost_samples = cost_dists.sample()

    return feats, cost_samples, cost_dist_params
