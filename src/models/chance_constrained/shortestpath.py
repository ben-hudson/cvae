import gurobipy as gp
import pyepo
import torch

from gurobipy import GRB
from typing import Tuple


class BinaryShortestPath(pyepo.model.grb.shortestPathModel):
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


class ChanceConstrainedShortestPath(BinaryShortestPath):
    def __init__(
        self, grid: Tuple[int], costs_mean: torch.Tensor, costs_std: torch.Tensor, budget: float, alpha: float
    ):
        self.costs_mean = torch.FloatTensor(costs_mean)
        self.costs_std = torch.FloatTensor(costs_std)
        self.budget = budget  # maximum allowable distance

        standard_normal = torch.distributions.Normal(0, 1)
        self.risk_tol = standard_normal.icdf(torch.tensor(alpha))

        super().__init__(grid)

    def _getModel(self):
        m, x = super()._getModel()

        obj = gp.quicksum(self.costs_mean[i] * x[k] for i, k in enumerate(x))
        m.setObjective(obj)

        # chance constraint
        # t is a dummy variable to track the upper bound on sqrt(x^T * cov * x)
        obj_std = m.addVar(name="t")
        # so, t^2 >= x^T * cov * x
        # cov is diagonal so x^T * cov * x = x * cov_ii * x = std^2 * x^2
        m.addConstr(obj_std**2 >= gp.quicksum((self.costs_std[i] ** 2) * (x[k] ** 2) for i, k in enumerate(x)))
        m.addConstr(obj + self.risk_tol * obj_std <= self.budget)

        return m, x

    def setObj(self, c):
        raise Exception("Setting objective isn't allowed because it changes the problem constraints")


class CVaRShortestPath(BinaryShortestPath):
    def __init__(self, grid: Tuple[int], costs_mean: torch.Tensor, costs_std: torch.Tensor, alpha: float):
        self.costs_mean = torch.FloatTensor(costs_mean)
        self.costs_std = torch.FloatTensor(costs_std)

        standard_normal = torch.distributions.Normal(0, 1)
        self.risk_tol = standard_normal.log_prob(standard_normal.icdf(torch.tensor(alpha))).exp() / (1 - alpha)

        super().__init__(grid)

    def _getModel(self):
        m, x = super()._getModel()

        obj = gp.quicksum(self.costs_mean[i] * x[k] for i, k in enumerate(x))

        # t is a dummy variable to track the upper bound on sqrt(x^T * cov * x)
        obj_std = m.addVar(name="t")
        # so, t^2 >= x^T * cov * x
        # cov is diagonal so x^T * cov * x = x * cov_ii * x = std^2 * x^2
        m.addConstr(obj_std**2 >= gp.quicksum((self.costs_std[i] ** 2) * (x[k] ** 2) for i, k in enumerate(x)))
        m.setObjective(obj + self.risk_tol * obj_std)

        return m, x

    def setObj(self, c):
        raise Exception("Setting objective isn't allowed because it changes the problem constraints")


class VaRShortestPath(BinaryShortestPath):
    def __init__(self, grid: Tuple[int], costs_mean: torch.Tensor, costs_std: torch.Tensor, alpha: float):
        self.costs_mean = torch.FloatTensor(costs_mean)
        self.costs_std = torch.FloatTensor(costs_std)

        standard_normal = torch.distributions.Normal(0, 1)
        self.risk_tol = standard_normal.icdf(torch.tensor(alpha))

        super().__init__(grid)

    def _getModel(self):
        m, x = super()._getModel()

        obj = gp.quicksum(self.costs_mean[i] * x[k] for i, k in enumerate(x))

        # t is a dummy variable to track the upper bound on sqrt(x^T * cov * x)
        obj_std = m.addVar(name="t")
        # so, t^2 >= x^T * cov * x
        # cov is diagonal so x^T * cov * x = x * cov_ii * x = std^2 * x^2
        m.addConstr(obj_std**2 >= gp.quicksum((self.costs_std[i] ** 2) * (x[k] ** 2) for i, k in enumerate(x)))
        m.setObjective(obj + self.risk_tol * obj_std)

        return m, x

    def setObj(self, c):
        raise Exception("Setting objective isn't allowed because it changes the problem constraints")
