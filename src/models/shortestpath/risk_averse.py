import torch
import gurobipy as gp

from networkx import DiGraph

from .risk_neutral import ILPShortestPath


class VaRShortestPath(ILPShortestPath):
    def __init__(self, graph: DiGraph, source, sink, risk_level: float) -> None:
        super().__init__(graph, source, sink)

        standard_normal = torch.distributions.Normal(0, 1)
        self.quantile = standard_normal.icdf(torch.tensor(risk_level))

        super().__init__(graph, source, sink)

    def setObj(self, cost_dist_params: torch.Tensor):
        costs_mean, costs_std = cost_dist_params.chunk(2, dim=-1)
        self._model, self.x = self._getModel()
        assert len(self.x) == len(costs_mean) == len(costs_std)

        # risk neutral objective
        obj = gp.quicksum(c * self.x[key] for c, key in zip(costs_mean, self.x))

        # the VaR formulation adds a penalty on the stdev of the solution, calculated by sqrt(x^T * cov * x)
        # we use a dummy variable and constraint to track this quantity
        obj_std = self._model.addVar(name="d")

        # therefore, d^2 >= x^T * cov * x
        # cov is diagonal so x^T * cov * x => x * cov_ii * x => std^2 * x^2
        self._model.addConstr(
            obj_std**2 >= gp.quicksum((c**2) * (self.x[key] ** 2) for c, key in zip(costs_std, self.x))
        )

        # finally, the objective is the risk neutral one plus a penalty on the sttdev
        self._model.setObjective(obj + self.quantile * obj_std)


class CVaRShortestPath(ILPShortestPath):
    def __init__(self, graph: DiGraph, source, sink, risk_level: float, tail: str = "right") -> None:
        super().__init__(graph, source, sink)

        # right tail CVaR_a = left tail CVaR_{1-a}
        # we flip alpha accordingly and calculate the right tail CVaR
        if tail == "right":
            self.beta = torch.tensor(risk_level)
        elif tail == "left":
            self.beta = torch.tensor(1 - risk_level)
        else:
            raise ValueError(f"tail must be 'left' or 'right'")

        standard_normal = torch.distributions.Normal(0, 1)
        quantile = standard_normal.icdf(self.beta)
        self.prob_quantile = standard_normal.log_prob(quantile).exp()

        super().__init__(graph, source, sink)

    def setObj(self, cost_dist_params):
        costs_mean, costs_std = cost_dist_params.chunk(2, dim=-1)
        self._model, self.x = self._getModel()
        assert len(self.x) == len(costs_mean) == len(costs_std)

        # objective formulation is almost identitcal to VaRShortestPath
        # see there for explanation
        obj = gp.quicksum(c * self.x[key] for c, key in zip(costs_mean, self.x))
        obj_std = self._model.addVar(name="d")
        self._model.addConstr(
            obj_std**2 >= gp.quicksum((c**2) * (self.x[key] ** 2) for c, key in zip(costs_std, self.x))
        )
        self._model.setObjective(obj + obj_std * self.prob_quantile / (1 - self.beta))


def get_obj_dist(sol: torch.Tensor, cost_dist_params: torch.Tensor):
    costs_mean, costs_std = cost_dist_params.chunk(2, dim=-1)
    costs_cov = torch.diag(costs_std**2)
    obj = torch.dot(costs_mean, sol)
    if sol.dim() == 1:
        obj_var = sol.unsqueeze(0) @ costs_cov @ sol.unsqueeze(1)
    elif sol.dim() == 2:
        obj_var = sol.T @ costs_cov @ sol
    else:
        raise Exception(f"expected sol to be a 1D or 2D tensor, but got a {sol.dim()}D one")
    return obj, obj_var.squeeze().sqrt()


def get_VaR(alpha: float, mu: float, sigma: float):
    quantile = torch.distributions.Normal(0, 1).icdf(torch.tensor(alpha))
    return mu + quantile * sigma


def get_CVaR(alpha: float, mu: float, sigma: float, tail="right"):
    # right tail CVaR_a = left tail CVaR_{1-a}
    # we flip alpha accordingly and calculate the right tail CVaR
    if tail == "right":
        beta = torch.tensor(alpha)
    elif tail == "left":
        beta = torch.tensor(1 - alpha)
    else:
        raise ValueError(f"tail must be 'left' or 'right'")
    standard_normal = torch.distributions.Normal(0, 1)
    quantile = standard_normal.icdf(beta)
    prob_quantile = standard_normal.log_prob(quantile).exp()

    # right tail CVaR_b = E[X | X >= VaR_b]
    # for a normal distribution (https://en.wikipedia.org/wiki/Expected_shortfall#Normal_distribution)
    return mu + sigma * prob_quantile / (1 - beta)
