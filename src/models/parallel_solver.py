import torch
import multiprocessing as mp


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
        sols = []
        if self.processes > 1:
            with mp.Pool(processes=self.processes) as pool:
                for sol in pool.imap(self._solve_instance, costs):
                    sols.append(sol)
                pool.close()
                pool.join()
        else:
            for cost in costs:
                sol = self._solve_instance(cost)
                sols.append(sol)

        sols = torch.stack(sols, dim=0)  # recombine into batch
        return sols.to(device)

    def _solve_instance(self, costs: torch.Tensor) -> torch.Tensor:
        model = self.model_cls(*self.model_args, **self.model_kwargs)
        model.setObj(costs)
        sol, _ = model.solve()
        if not isinstance(sol, torch.Tensor):
            sol = torch.FloatTensor(sol)
        return sol
