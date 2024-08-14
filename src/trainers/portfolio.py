import pyepo
import torch
import torch.nn.functional as F

from cooper import ConstrainedMinimizationProblem as CMP, CMPState
from metrics import r2, mcc
from pyepo.metric import calRegret
from torch.distributions import kl_divergence


class PortfolioTrainer(CMP):
    def __init__(self, unnorm_func, device, portfolio_model, costs_are_latents=False, kld_weight=1):
        super().__init__(is_constrained=True)

        self.unnorm = unnorm_func
        self.device = device
        self.portfolio_model = portfolio_model
        self.costs_are_latents = costs_are_latents
        self.kld_weight = kld_weight

    def closure(self, model, batch):
        feats_normed, costs_normed, sols_normed, objs_normed = batch
        feats_normed = feats_normed.to(self.device)
        costs_normed = costs_normed.to(self.device)
        sols_normed = sols_normed.to(self.device)
        objs_normed = objs_normed.to(self.device)

        if self.costs_are_latents:
            prior, posterior, costs_hat_normed, sols_hat_normed = model(feats_normed, sols_normed)
        else:
            obs_normed = torch.cat([costs_normed, sols_normed], dim=-1)
            prior, posterior, latents_hat, obs_hat = model(feats_normed, obs_normed)
            costs_hat_normed, sols_hat_normed = obs_hat.split([costs_normed.size(-1), sols_normed.size(-1)], dim=-1)

        _, costs_hat, sols_hat, _ = self.unnorm(None, costs_hat_normed, sols_hat_normed, None)
        feats, costs, sols, objs = self.unnorm(feats_normed, costs_normed, sols_normed, objs_normed)

        kld = kl_divergence(prior, posterior).mean()
        mse = F.mse_loss(costs_hat_normed, costs_normed, reduce="mean")
        loss = self.kld_weight * kld + (1 - self.kld_weight) * mse

        sols = sols.unsqueeze(2)
        costs = costs.unsqueeze(1)
        sols_hat = sols_hat.unsqueeze(2)
        costs_hat = costs_hat.unsqueeze(1)
        objs_hat = torch.bmm(costs_hat, sols_hat).squeeze(2)

        regret = torch.bmm(costs, sols_hat).squeeze(2) - objs
        satisfaction = torch.bmm(costs_hat, sols).squeeze(2) - objs  # the opposite of regret, haha

        eq_constrs = torch.stack(
            [
                objs_hat - objs,
                regret,
                satisfaction,
            ],
            dim=1,
        )

        # regret is the other way around when maximizing
        if self.portfolio_model.modelSense == pyepo.EPO.MAXIMIZE:
            eq_constrs *= -1

        budget = 1  # budget is hardcoded to 1
        budget_constr = sols_hat.sum(axis=1) - budget

        cov = (
            torch.FloatTensor(self.portfolio_model.covariance).to(self.device).expand(sols_hat.size(0), -1, -1)
        )  # create batch of covariance matrices
        risk_constr = (
            torch.bmm(sols_hat.permute(0, 2, 1), torch.bmm(cov, sols_hat)).squeeze(2) - self.portfolio_model.risk_level
        )

        ineq_constrs = torch.stack(
            [
                budget_constr,
                risk_constr,
            ],
            dim=1,
        )

        metrics = {
            "loss": loss.detach(),
            "kld": kld.detach(),
            "cost_mse": mse.detach(),
            "regret": regret.detach().mean(),
            "satisfaction": satisfaction.detach().mean(),
            "budget_viol": budget_constr.detach().clamp(min=0).mean(),
            "risk_viol": risk_constr.detach().clamp(min=0).mean(),
        }

        return CMPState(loss=loss, ineq_defect=ineq_constrs, eq_defect=eq_constrs, misc=metrics)

    def val(self, model, batch):
        feats_normed, costs_normed, sols_normed, objs_normed = batch
        latents_hat, obs_hat = model.sample(feats_normed.to(self.device), mean=True)

        if self.costs_are_latents:
            costs_hat_normed = latents_hat
        else:
            costs_hat_normed, _ = obs_hat.split([costs_normed.size(-1), sols_normed.size(-1)], dim=-1)
        costs_hat_normed = costs_hat_normed.cpu()

        _, costs_hat, _, _ = self.unnorm(costs_normed=costs_hat_normed)
        _, costs, _, objs = self.unnorm(costs_normed=costs_normed, objs_normed=objs_normed)

        total_regret = 0
        total_obj = 0
        for i in range(costs.size(-1)):
            obj = objs[i].item()
            total_regret += calRegret(self.portfolio_model, costs_hat[i], costs[i], obj)
            total_obj += abs(obj)

        metrics = {
            "mcc": mcc(costs_hat_normed, costs_normed),
            "r2": r2(costs_hat_normed, costs_normed),
            "total_regret": total_regret,
            "total_obj": total_obj,
        }

        return metrics
