import torch
import multiprocessing as mp


class ParallelSolver(torch.nn.Module):
    def __init__(self, processes, model_cls, *model_args, **model_kwargs):
        super().__init__()

        self.processes = processes

        self.model_cls = model_cls
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def forward(self, costs_pred: torch.Tensor):
        device = costs_pred.device
        costs_pred = costs_pred.detach().cpu()

        # solve
        if self.processes > 1:
            with mp.Pool(processes=self.processes) as pool:
                chunk_size = len(costs_pred) // self.processes
                sols = pool.map(self._solve, costs_pred, chunk_size)
        else:
            sols = self._solve(costs_pred)

        # convert to tensor
        sols = torch.FloatTensor(sols).to(device)
        return sols

    def _solve(self, costs):
        model = self.model_cls(*self.model_args, **self.model_kwargs)

        sols = []
        for cost in costs:
            model.setObj(cost)
            sol, _ = model.solve()
            sols.append(sol)
        return sols
