import torch
import pytest

from distributions import TwoPoint, expectation, variance
from models.parallel_solver import ParallelSolver
from models.shortestpath.risk_averse import CVaRShortestPath
from models.shortestpath.risk_neutral import ILPShortestPath

from data.toy_example import get_toy_graph, gen_toy_data


@pytest.fixture
def toy_graph():
    return get_toy_graph()


@pytest.fixture
def toy_data():
    return gen_toy_data(1000)


def test_wait_and_see_policy(toy_graph, toy_data):
    _, costs, _ = toy_data
    solver = ParallelSolver(1, ILPShortestPath, toy_graph, 0, 1)

    sols = solver(costs)
    objs = torch.bmm(costs.unsqueeze(1), sols.unsqueeze(2)).squeeze()

    # the wait-and-see policy takes the random edge when it is short (80% of the time)
    # and it takes the deterministic edge when the random edge is long (20% of the time)
    # therefore, the expected objective is 5*0.8 + 10*0.2 = 6
    assert torch.isclose(objs.mean(), torch.tensor(6.0), rtol=0.1)
    assert torch.isclose(objs.var(), torch.tensor(4.0), rtol=0.1)


def test_risk_neutral_policy(toy_graph, toy_data):
    _, costs, cost_dist_params = toy_data
    solver = ParallelSolver(1, ILPShortestPath, toy_graph, 0, 1)

    cost_lows, cost_highs, cost_probs = cost_dist_params.chunk(3, dim=-1)
    cost_dists = TwoPoint(cost_lows, cost_highs, cost_probs)

    cost_means = expectation(cost_dists)
    sols = solver(cost_means)
    objs = torch.bmm(costs.unsqueeze(1), sols.unsqueeze(2)).squeeze()

    # the risk neutral policy takes the random edge every time because the expectation is lower
    # therefore, the expected objective is the same as the random edge, 8
    assert torch.isclose(objs.mean(), torch.tensor(8.0), rtol=0.1)
    assert torch.isclose(objs.var(), torch.tensor(36.0), rtol=0.1)


def test_risk_averse_policy(toy_graph, toy_data):
    _, costs, cost_dist_params = toy_data
    solver = ParallelSolver(1, CVaRShortestPath, toy_graph, 0, 1, 0.9, tail="right")

    cost_lows, cost_highs, cost_probs = cost_dist_params.chunk(3, dim=-1)
    cost_dists = TwoPoint(cost_lows, cost_highs, cost_probs)

    cost_means = expectation(cost_dists)
    cost_stds = variance(cost_dists).sqrt()
    sols = solver(torch.cat([cost_means, cost_stds], dim=-1))
    objs = torch.bmm(costs.unsqueeze(1), sols.unsqueeze(2)).squeeze()

    # the risk averse policy takes the deterministic edge every time because it is risk averse
    # therefore, the expected objective is the same as the deterministic edge, 10
    assert torch.isclose(objs.mean(), torch.tensor(10.0), rtol=0.1)
    assert torch.isclose(objs.var(), torch.tensor(0.0), rtol=0.1)
