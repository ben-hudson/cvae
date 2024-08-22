import gurobipy as gp
import pyepo
import torch


class ChanceConstrainedShortestPath(pyepo.model.grb.shortestPathModel):
    def __init__(self, grid, costs_mean, costs_std, budget, prob_thresh):
        self.costs = torch.distributions.Normal(costs_mean, costs_std)
        self.budget = budget  # maximum allowable distance

        standard_normal = torch.distributions.Normal(0, 1)
        self.thresh = standard_normal.icdf(torch.tensor(prob_thresh))

        super().__init__(grid)

    def _getModel(self):
        m, x = super()._getModel()

        obj = gp.quicksum(self.costs.loc[i] * x[k] for i, k in enumerate(x))
        m.setObjective(obj)

        # add chance constraint
        # t represents an upper bound on sqrt(x^T * cov * x)
        t = m.addVar(name="t")
        # so, t^2 >= x^T * cov * x
        # since cov is diagonal, x^T * cov * x = cov_ii * x^2 = std^2 * x^2
        m.addQConstr(t**2 >= gp.quicksum((self.costs.scale[i] ** 2) * (x[k] ** 2) for i, k in enumerate(x)))
        m.addConstr(obj + self.thresh * t <= self.budget)

        return m, x

    def setObj(self, c):
        raise Exception("Setting objective isn't allowed because it changes the constraints")

    def solve(self):
        try:
            return super().solve()
        except AttributeError as e:
            if self._model.status != gp.GRB.OPTIMAL:
                raise Exception(f"Solve failed with status code {self._model.status}")
            # otherwise, raise the original error
            raise e
