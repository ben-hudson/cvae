import gurobipy as gp
import pyepo
import torch

from gurobipy import GRB
from typing import Tuple


class RiskAverseShortestPath(pyepo.model.grb.shortestPathModel):
    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("shortest path")
        # varibles
        x = m.addVars(self.arcs, name="x", vtype=GRB.BINARY)
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= x[e]
                # source
                if i == 0 and j == 0:
                    m.addConstr(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                    m.addConstr(expr == 1)
                # transition
                else:
                    m.addConstr(expr == 0)
        return m, x

    def solve(self):
        try:
            return super().solve()
        except AttributeError as e:
            if self._model.status != gp.GRB.OPTIMAL:
                raise Exception(f"Solve failed with status code {self._model.status}")
            # otherwise, raise the original error
            raise e


class CVaRShortestPath(RiskAverseShortestPath):
    def __init__(self, grid: Tuple[int], alpha: float, tail: str = "right"):

        # right tail CVaR_a = left tail CVaR_{1-a}
        # we flip alpha accordingly and calculate the right tail CVaR
        if tail == "right":
            self.beta = torch.tensor(alpha)
        elif tail == "left":
            self.beta = torch.tensor(1 - alpha)
        else:
            raise ValueError(f"tail must be 'left' or 'right'")

        standard_normal = torch.distributions.Normal(0, 1)
        quantile = standard_normal.icdf(self.beta)
        self.prob_quantile = standard_normal.log_prob(quantile).exp()

        super().__init__(grid)

    def setObj(self, cost_dist_params):
        costs_mean, costs_std = cost_dist_params.chunk(2, dim=-1)

        m, x = self._getModel()  # get a fresh model
        obj = gp.quicksum(costs_mean[i] * x[k] for i, k in enumerate(x))

        # s is a dummy variable to track the upper bound on sqrt(x^T * cov * x)
        obj_std = m.addVar(name="s")
        # so, s^2 >= x^T * cov * x
        # cov is diagonal so x^T * cov * x = x * cov_ii * x = std^2 * x^2
        m.addConstr(obj_std**2 >= gp.quicksum((costs_std[i] ** 2) * (x[k] ** 2) for i, k in enumerate(x)))
        m.setObjective(obj + obj_std * self.prob_quantile / (1 - self.beta))

        self._model, self.x = m, x


class VaRShortestPath(RiskAverseShortestPath):
    def __init__(self, grid: Tuple[int], alpha: float):
        standard_normal = torch.distributions.Normal(0, 1)
        self.quantile = standard_normal.icdf(torch.tensor(alpha))

        super().__init__(grid)

    def setObj(self, cost_dist_params):
        costs_mean, costs_std = cost_dist_params.chunk(2, dim=-1)

        m, x = super()._getModel()  # get a fresh model
        obj = gp.quicksum(costs_mean[i] * x[k] for i, k in enumerate(x))

        # s is a dummy variable to track the upper bound on sqrt(x^T * cov * x)
        obj_std = m.addVar(name="s")
        # so, s^2 >= x^T * cov * x
        # cov is diagonal so x^T * cov * x = x * cov_ii * x = std^2 * x^2
        m.addConstr(obj_std**2 >= gp.quicksum((costs_std[i] ** 2) * (x[k] ** 2) for i, k in enumerate(x)))
        m.setObjective(obj + self.quantile * obj_std)

        self._model, self.x = m, x
