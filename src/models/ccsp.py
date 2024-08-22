import gurobipy as gp
import pyepo
import torch

from typing import Tuple


class ChanceConstrainedShortestPath(pyepo.model.grb.shortestPathModel):
    def __init__(
        self, grid: Tuple[int], costs_mean: torch.Tensor, costs_std: torch.Tensor, budget: float, alpha: float
    ):
        self.costs_mean = costs_mean
        self.costs_std = costs_std
        self.budget = budget  # maximum allowable distance

        assert alpha >= 0.5, "alpha must be at least 0.5 (0.5 corresponds to a risk-neutral decision)"
        standard_normal = torch.distributions.Normal(0, 1)
        self.thresh = standard_normal.icdf(torch.tensor(alpha))

        super().__init__(grid)

    def _getModel(self):
        m, x = super()._getModel()

        obj = gp.quicksum(self.costs_mean[i] * x[k] for i, k in enumerate(x))
        m.setObjective(obj)

        # add chance constraint
        # t is a dummy variable to track the upper bound on sqrt(x^T * cov * x)
        t = m.addVar(name="t")
        # so, t^2 >= x^T * cov * x
        # cov is diagonal so x^T * cov * x = x * cov_ii * x = std^2 * x^2
        m.addQConstr(t**2 >= gp.quicksum((self.costs_std[i] ** 2) * (x[k] ** 2) for i, k in enumerate(x)))
        m.addConstr(obj + self.thresh * t <= self.budget)

        return m, x

    def setObj(self, c):
        raise Exception("Setting objective isn't allowed because it changes the problem constraints")

    def solve(self):
        try:
            return super().solve()
        except AttributeError as e:
            if self._model.status != gp.GRB.OPTIMAL:
                raise Exception(f"Solve failed with status code {self._model.status}")
            # otherwise, raise the original error
            raise e
