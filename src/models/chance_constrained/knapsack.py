import gurobipy as gp
import pyepo
import torch


class ChanceConstrainedKnapsack(pyepo.model.grb.knapsackModel):
    def __init__(
        self,
        weights: torch.Tensor,
        capacity: torch.Tensor,
        costs_mean: torch.Tensor,
        costs_std: torch.Tensor,
        min_obj: float,
        prob: float,
    ):
        self.costs_mean = costs_mean
        self.costs_std = costs_std
        self.min_obj = min_obj

        # the chance constraint is that the objective is LESS than the constraint level
        # so when maximizing, we want this to be a small probability
        # P(obj < min_obj) <= (1 - prob)
        standard_normal = torch.distributions.Normal(0, 1)
        self.thresh = standard_normal.icdf(torch.tensor(1 - prob))

        super().__init__(weights, capacity)

    def _getModel(self):
        m, x = super()._getModel()

        obj = self.costs_mean.numpy() @ x
        m.setObjective(obj)

        # add chance constraint
        # t is a dummy variable to track the upper bound on sqrt(x^T * cov * x)
        t = m.addVar(name="t")
        # so, t^2 >= x^T * cov * x
        cov = torch.diag(self.costs_std**2).numpy()
        m.addQConstr(t**2 >= x.T @ cov @ x)
        m.addConstr(obj + self.thresh * t <= self.min_obj)

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


if __name__ == "__main__":
    pass
