import networkx as nx
import numpy as np
import pytest
import torch

from models.shortestpath.risk_neutral import ShortestPath, ILPShortestPath


@pytest.fixture
def random_graph():
    G = nx.fast_gnp_random_graph(10, 0.3, directed=True)
    node_pos = nx.spring_layout(G)
    for i, j in G.edges:
        G.edges[i, j]["cost"] = np.linalg.norm(node_pos[j] - node_pos[i])
    return G


def path_to_sol(path, edges):
    path_edges = list(zip(path[:-1], path[1:]))
    sol = [1 if e in path_edges else 0 for e in edges]
    return torch.FloatTensor(sol)


def test_shortestpath(random_graph: nx.DiGraph):
    node_list = list(random_graph.nodes)
    source = node_list[0]
    sink = node_list[-1]

    true_obj, true_path = nx.single_source_bellman_ford(random_graph, source, sink, "cost")
    true_sol = path_to_sol(true_path, random_graph.edges)

    cost_dict = nx.get_edge_attributes(random_graph, "cost")
    cost = torch.FloatTensor(list(cost_dict.values()))

    model = ShortestPath(random_graph, source, sink)
    model.setObj(cost)
    sol, obj = model.solve()

    assert np.isclose(obj, true_obj), f"model and true objective values differ"
    assert (sol == true_sol).all(), f"model and true paths differ"


def test_ilp_shortestpath(random_graph: nx.DiGraph):
    node_list = list(random_graph.nodes)
    source = node_list[0]
    sink = node_list[-1]

    true_obj, true_path = nx.single_source_bellman_ford(random_graph, source, sink, "cost")
    true_sol = path_to_sol(true_path, random_graph.edges)

    cost_dict = nx.get_edge_attributes(random_graph, "cost")
    cost = torch.FloatTensor(list(cost_dict.values()))

    model = ILPShortestPath(random_graph, source, sink)
    model.setObj(cost)
    sol, obj = model.solve()

    assert np.isclose(obj, true_obj), f"model and true objective values differ"
    assert (sol == true_sol).all(), f"model and true paths differ"
