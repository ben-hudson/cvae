import networkx as nx
import numpy as np
import pytest
import torch

from models.shortestpath.risk_neutral import ShortestPath
from models.shortestpath.risk_averse import VaRShortestPath, CVaRShortestPath, get_obj_dist, get_VaR, get_CVaR


@pytest.fixture
def random_graph():
    G = nx.fast_gnp_random_graph(10, 0.3, directed=True)
    node_pos = nx.spring_layout(G)
    for i, j in G.edges:
        G.edges[i, j]["cost"] = np.linalg.norm(node_pos[j] - node_pos[i])
    return G


@pytest.fixture
def risk_level():
    return 0.9


def test_VaR_shortestpath(random_graph: nx.DiGraph, risk_level: float):
    node_list = list(random_graph.nodes)
    source = node_list[0]
    sink = node_list[-1]
    cost_dict = nx.get_edge_attributes(random_graph, "cost")
    cost_mean = torch.FloatTensor(list(cost_dict.values()))

    risk_neutral_model = ShortestPath(random_graph, source, sink)
    risk_neutral_model.setObj(cost_mean)
    risk_neutral_sol, risk_neutral_obj = risk_neutral_model.solve()

    # set the cost stds to a low value, except along the risk netural path
    cost_std = 0.1 * torch.ones_like(cost_mean)
    cost_std[risk_neutral_sol == 1] = 10.0
    cost_dist_params = torch.cat([cost_mean, cost_std], dim=-1)

    risk_averse_model = VaRShortestPath(random_graph, source, sink, risk_level)
    risk_averse_model.setObj(cost_dist_params)
    risk_averse_sol, risk_averse_obj = risk_averse_model.solve()

    risk_neutral_obj_mean, risk_neutral_obj_std = get_obj_dist(risk_neutral_sol, cost_dist_params)
    risk_averse_obj_mean, risk_averse_obj_std = get_obj_dist(risk_averse_sol, cost_dist_params)
    risk_neutral_VaR = get_VaR(risk_level, risk_neutral_obj_mean, risk_neutral_obj_std)
    risk_averse_VaR = get_VaR(risk_level, risk_averse_obj_mean, risk_averse_obj_std)

    assert np.isclose(risk_neutral_obj_mean, risk_neutral_obj)
    assert np.isclose(risk_averse_obj, risk_averse_VaR, rtol=0.001)
    assert risk_neutral_VaR >= risk_averse_VaR


def test_CVaR_shortestpath(random_graph: nx.DiGraph, risk_level: float):
    node_list = list(random_graph.nodes)
    source = node_list[0]
    sink = node_list[-1]
    cost_dict = nx.get_edge_attributes(random_graph, "cost")
    cost_mean = torch.FloatTensor(list(cost_dict.values()))

    risk_neutral_model = ShortestPath(random_graph, source, sink)
    risk_neutral_model.setObj(cost_mean)
    risk_neutral_sol, risk_neutral_obj = risk_neutral_model.solve()

    # set the cost stds to a low value, except along the risk netural path
    cost_std = 0.1 * torch.ones_like(cost_mean)
    cost_std[risk_neutral_sol == 1] = 10.0
    cost_dist_params = torch.cat([cost_mean, cost_std], dim=-1)

    risk_averse_model = CVaRShortestPath(random_graph, source, sink, risk_level, tail="right")
    risk_averse_model.setObj(cost_dist_params)
    risk_averse_sol, risk_averse_obj = risk_averse_model.solve()

    risk_neutral_obj_mean, risk_neutral_obj_std = get_obj_dist(risk_neutral_sol, cost_dist_params)
    risk_averse_obj_mean, risk_averse_obj_std = get_obj_dist(risk_averse_sol, cost_dist_params)
    risk_neutral_CVaR = get_CVaR(risk_level, risk_neutral_obj_mean, risk_neutral_obj_std, tail="right")
    risk_averse_CVaR = get_CVaR(risk_level, risk_averse_obj_mean, risk_averse_obj_std, tail="right")

    assert np.isclose(risk_neutral_obj_mean, risk_neutral_obj)
    assert np.isclose(risk_averse_obj, risk_averse_CVaR)
    assert risk_neutral_CVaR >= risk_averse_CVaR
