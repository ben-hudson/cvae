import gurobipy as gp
import networkx as nx
import torch

from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel
from typing import Tuple

from utils.utils import hush


# bellman ford SP problem (we use bellman ford to allow for negative edge costs)
class ShortestPath:
    def __init__(self, graph: nx.DiGraph, source, sink) -> None:
        super().__init__()

        self._graph = graph
        self._source = source
        self._sink = sink

        assert self._graph.is_directed, f"the graph should be directed"
        assert self._source in self._graph.nodes, f"source node {self._source} does not exist in the graph"
        assert self._sink in self._graph.nodes, f"sink node {self._sink} does not exist in the graph"

    def setObj(self, costs: torch.Tensor) -> None:
        assert len(self._graph.edges) == len(costs)
        for e, val in zip(self._graph.edges, costs):
            self._graph.edges[e]["cost"] = val

    def solve(self) -> Tuple[torch.FloatTensor, float]:
        obj, path = nx.single_source_bellman_ford(self._graph, self._source, self._sink, "cost")
        path_edges = list(zip(path[:-1], path[1:]))
        sol = [1 if e in path_edges else 0 for e in self._graph.edges]
        return torch.FloatTensor(sol), obj


# an ILP formulation of the SP problem
# unfortunately, we have to stick to the PyEPO formula so it is compatible with other parts of the package
class ILPShortestPath(optGrbModel):
    def __init__(self, graph: nx.DiGraph, source, sink) -> None:
        self._graph = graph
        self._source = source
        self._sink = sink

        assert self._graph.is_directed, f"the graph should be directed"
        assert self._source in self._graph.nodes, f"source node {self._source} does not exist in the graph"
        assert self._sink in self._graph.nodes, f"sink node {self._sink} does not exist in the graph"

        super().__init__()  # sets self._model and self.x via _getModel()

    def setObj(self, costs: torch.Tensor) -> None:
        assert len(self.x) == len(costs)
        obj = gp.quicksum(c * self.x[key] for c, key in zip(costs, self.x))
        self._model.setObjective(obj)

    def solve(self) -> Tuple[torch.FloatTensor, float]:
        with hush():
            sol, obj = super().solve()
        if self._model.Status != GRB.OPTIMAL:
            raise Exception(f"Solve failed with status code {self._model.Status}")

        return torch.FloatTensor(sol), obj

    def _getModel(self) -> Tuple[gp.Model, gp.Var]:
        model = gp.Model("SP")
        model.ModelSense = GRB.MINIMIZE

        # returns a dict-like object where the edges are keys
        x = model.addVars(self._graph.edges, name="x", vtype=GRB.BINARY)

        # flow conservation on each node
        for j in self._graph.nodes:
            expr = 0
            # flow in
            for i in self._graph.predecessors(j):
                expr += x[i, j]
            # flow out
            for k in self._graph.successors(j):
                expr -= x[j, k]
            if j == self._source:
                model.addConstr(expr == -1)
            elif j == self._sink:
                model.addConstr(expr == 1)
            else:
                model.addConstr(expr == 0)

        return model, x
