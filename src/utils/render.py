import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from matplotlib.figure import Figure


def render_graph(graph: nx.Graph, edge_color: torch.Tensor, cmap_name: str = "viridis"):
    fig, ax = plt.subplots(figsize=(1, 1))
    cmap = plt.get_cmap(cmap_name)

    assert len(edge_color) == len(
        graph.edges
    ), f"expected {len(graph.edges)} elements in edge_color but got {len(edge_color)}"

    node_pos = nx.get_node_attributes(graph, "pos")
    if len(node_pos) < len(graph.nodes):
        node_pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos=node_pos, node_size=25, node_color="grey", ax=ax)

    edge_color_normed = edge_color / edge_color.norm()
    for edge, color in zip(graph.edges, edge_color_normed):
        bend = edge[2] if graph.is_multigraph() else 0
        nx.draw_networkx_edges(
            graph, node_pos, edgelist=[edge], edge_color=cmap(color), connectionstyle=f"arc3,rad={bend*0.3}"
        )

    img = fig_to_rgb_tensor(fig)
    plt.close(fig)
    return img


def fig_to_rgb_tensor(fig: Figure):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_np = data.reshape((int(h), int(w), -1)).copy()
    # matplotlib puts channels last, pytorch puts channels first
    img_pt = torch.FloatTensor(img_np).permute(2, 0, 1)
    return img_pt
