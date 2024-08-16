import pyepo
import torch
import torch.nn.functional as F

from cooper import ConstrainedMinimizationProblem as CMP, CMPState
from metrics import r2, mcc
from pyepo.metric import calRegret
from torch.distributions import kl_divergence


class LPTrainer(CMP):
    def __init__(self, unnorm_func, device, lp, is_integer=False, costs_are_latents=False, kld_weight=1):
        super().__init__(is_constrained=True)

        self.unnorm = unnorm_func
        self.device = device
        self.lp = lp
        self.is_integer = is_integer
        self.costs_are_latents = costs_are_latents
        self.kld_weight = kld_weight

    def get_constrs(self):
        n_constrs = self.lp._model.NumConstrs
        n_vars = self.lp._model.NumVars

        A_eq = torch.zeros((n_constrs, n_vars))
        b_eq = torch.zeros(n_constrs)

        for i, constr in enumerate(self.lp._model.getConstrs()):
            if constr.Sense == "=":
                b_eq[i] = constr.RHS
                for j, var in enumerate(self.lp._model.getVars()):
                    A_eq[i, j] = self.lp._model.getCoeff(constr, var)

            else:
                raise ValueError(f"unhandled constraint sense: {constr.Sense}")

        return A_eq, b_eq

    def closure(self, model, batch):
        feats_normed, costs_normed, sols_normed, objs_normed = batch
        feats_normed = feats_normed.to(self.device)
        costs_normed = costs_normed.to(self.device)
        sols_normed = sols_normed.to(self.device)
        objs_normed = objs_normed.to(self.device)

        feats, costs, sols, objs = self.unnorm(feats_normed, costs_normed, sols_normed, objs_normed)

        if self.costs_are_latents:
            prior, posterior, costs_pred_normed, sols_pred_normed = model(feats_normed, sols_normed)
        else:
            obs_normed = torch.cat([costs_normed, sols_normed], dim=-1)
            prior, posterior, latents_hat, obs_hat = model(feats_normed, obs_normed)
            costs_pred_normed, sols_pred_normed = obs_hat.split([costs_normed.size(-1), sols_normed.size(-1)], dim=-1)

        if self.is_integer:
            sols_pred = F.sigmoid(sols_pred_normed)
            _, costs_pred, _, _ = self.unnorm(None, costs_pred_normed, None, None)
            decision_err = F.binary_cross_entropy(sols_pred, sols, reduction="mean")
        else:
            _, costs_pred, sols_pred, _ = self.unnorm(None, costs_pred_normed, sols_pred_normed, None)
            decision_err = F.mse_loss(sols_pred, sols, reduction="mean")

        kld = kl_divergence(prior, posterior).mean()
        cost_err = F.mse_loss(costs_pred_normed, costs_normed, reduction="mean")

        sols = sols.unsqueeze(2)
        costs = costs.unsqueeze(1)
        sols_pred = sols_pred.unsqueeze(2)
        costs_pred = costs_pred.unsqueeze(1)

        obj_pred = torch.bmm(costs_pred, sols_pred).squeeze(2)
        obj_realized = torch.bmm(costs, sols_pred).squeeze(2)

        # "true" SPO loss
        spo = obj_realized - objs

        # intuitively, regret is the difference between the expected cost and the realized cost
        # regret is positive when the realized cost exceeds the expected cost
        regret = obj_realized - obj_pred

        # these definitions flip when I am maximizing
        if self.lp.modelSense == pyepo.EPO.MAXIMIZE:
            spo *= -1
            regret *= -1

        A_eq, b_eq = self.get_constrs()
        A_eq_batch = A_eq.expand(sols_pred.size(0), -1, -1).to(self.device)
        b_eq_batch = b_eq.unsqueeze(1).to(self.device)
        # violation of the problem equality constraints
        lp_eq_constr = (torch.bmm(A_eq_batch, sols_pred) - b_eq_batch).squeeze(2)

        eq_constrs = torch.cat([spo, regret, lp_eq_constr], dim=-1)

        loss = kld

        metrics = {
            "cost_err": cost_err.detach(),
            "decision_err": decision_err.detach(),
            "kld": kld.detach(),
            "loss": loss.detach(),
            "lp_eq_constr_viol": lp_eq_constr.detach().mean(),
            "regret": regret.detach().mean(),
            "spo_loss": spo.detach().mean(),
            "total_obj": objs.detach().abs().sum(),
            "total_pyepo_regret": spo.detach().sum(),
        }

        return CMPState(loss=loss, ineq_defect=None, eq_defect=eq_constrs, misc=metrics)

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
            total_regret += calRegret(self.lp, costs_hat[i], costs[i], obj)
            total_obj += abs(obj)

        metrics = {
            "mcc": mcc(costs_hat_normed, costs_normed),
            "r2": r2(costs_hat_normed, costs_normed),
            "total_obj": total_obj,
            "total_pyepo_regret": total_regret,
        }

        return metrics
