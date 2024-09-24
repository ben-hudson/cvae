import torch
import multiprocessing as mp

from utils.utils import quiet


class ParallelSolver(torch.nn.Module):
    def __init__(self, processes, model_cls, *model_args, **model_kwargs):
        super().__init__()

        self.processes = processes

        self.model_cls = model_cls
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def forward(self, costs: torch.Tensor):
        device = costs.device
        costs = costs.detach().cpu()

        # solve
        if self.processes > 1:
            sols = []
            with mp.Pool(processes=self.processes) as pool:
                for sol in pool.imap(self._solve_instance, costs):
                    sols.append(sol)
                pool.close()
                pool.join()
        else:
            sols = self._solve(costs)

        sols = torch.FloatTensor(sols)
        sols = (sols > 0.5).float().to(device)

        return sols

    def _solve_instance(self, costs):
        model = self.model_cls(*self.model_args, **self.model_kwargs)
        model.setObj(costs)
        with quiet():
            sol, _ = model.solve()
        return sol

    def _solve(self, costs):
        model = self.model_cls(*self.model_args, **self.model_kwargs)

        sols = []
        for cost in costs:
            model.setObj(cost)
            with quiet():
                sol, _ = model.solve()
            sols.append(sol)
        return sols


if __name__ == "__main__":
    import pyepo
    import pyepo.data
    import pyepo.model.grb
    import torch.utils.data

    n_samples = 1000
    n_features = 5
    degree = 6
    seed = 42
    noise_width = 0.5

    grid = (5, 5)
    feats, costs_expected = pyepo.data.shortestpath.genData(
        n_samples, n_features, grid, deg=degree, noise_width=0, seed=seed
    )

    feats = torch.FloatTensor(feats)
    costs_expected = torch.FloatTensor(costs_expected)

    costs_std = noise_width / costs_expected.abs()

    cost_dist = "normal"
    cost_dist_params = torch.cat([costs_expected, costs_std], dim=-1)
    costs = torch.distributions.Normal(costs_expected, costs_std).sample()

    data_model = pyepo.model.grb.shortestPathModel(grid)

    dataset = pyepo.data.dataset.optDataset(data_model, feats, costs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=False, num_workers=1, drop_last=True)

    processes = 8
    parallel_solver = ParallelSolver(processes, pyepo.model.grb.shortestPathModel, grid)

    for feats, costs, sols, objs in loader:
        sols_pred = parallel_solver(costs)
        assert (sols_pred == sols).all()
