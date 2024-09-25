import pyepo.data
import pyepo.model.grb
import pytest
import torch.utils.data

from models.parallel_solver import ParallelSolver


@pytest.mark.parametrize("processes", [1, 4], ids=["singlethreaded", "multithreaded"])
def test_parallel_solver(processes):
    n_samples = 200
    n_features = 5
    degree = 6
    seed = 42
    noise_width = 0.5

    grid = (5, 5)
    feats, costs_mean = pyepo.data.shortestpath.genData(
        n_samples, n_features, grid, deg=degree, noise_width=0, seed=seed
    )

    feats = torch.FloatTensor(feats)

    costs_mean = torch.FloatTensor(costs_mean)
    costs_std = noise_width / costs_mean.abs()
    costs = torch.distributions.Normal(costs_mean, costs_std).sample()

    data_model = pyepo.model.grb.shortestPathModel(grid)

    dataset = pyepo.data.dataset.optDataset(data_model, feats, costs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(200, n_samples), shuffle=False, num_workers=1)

    parallel_solver = ParallelSolver(processes, pyepo.model.grb.shortestPathModel, grid)

    for feats, costs, sols, objs in loader:
        sols_pred = parallel_solver(costs)
        assert (sols_pred == sols).all()
