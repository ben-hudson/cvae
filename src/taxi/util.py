import networkx as nx
import numpy as np

from scipy.spatial import Delaunay
from scipy.stats.qmc import LatinHypercube

def generate_random_network(n_nodes):
    graph = nx.Graph()

    node_pos = LatinHypercube(2).random(n_nodes)
    for n, pos in enumerate(node_pos):
        graph.add_node(n, pos=pos)

    triangulation = Delaunay(node_pos)
    for tri in triangulation.simplices:
        graph.add_edge(tri[0], tri[1])
        graph.add_edge(tri[1], tri[2])
        graph.add_edge(tri[2], tri[0])

    for i, j in graph.edges:
        pos_i = graph.nodes[i]['pos']
        pos_j = graph.nodes[j]['pos']
        graph.edges[i, j]['cost'] = np.linalg.norm(pos_i - pos_j)

    return graph
