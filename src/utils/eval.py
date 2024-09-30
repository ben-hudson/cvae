import pyepo
import torch
import torch.nn.functional as F
import torch.distributions as D

from collections import namedtuple
from typing import Sequence

from utils.utils import norm, is_integer
from distributions import expectation, variance

EvalMetrics = namedtuple(
    "EvalMetrics",
    [
        "cost_err",
        "decision_err",
        "regret",
        "disappointment",
        "surprise",
        "obj_true",
        "obj_pred",
        "obj_expected",
        "obj_realized",
        "cost_dist_exp_err",
        "cost_dist_var_err",
    ],
)


def get_eval_metrics(
    costs_true: torch.Tensor,
    sols_true: torch.Tensor,
    cost_dists_true: D.Distribution,
    costs_pred: torch.Tensor,
    sols_pred: torch.Tensor,
    cost_dists_pred: D.Distribution,
    model_sense: int = pyepo.EPO.MINIMIZE,
    class_weights: Sequence = None,
):
    costs_true, _ = norm(costs_true)
    costs_pred, _ = norm(costs_pred)

    assert is_integer(sols_true), "we have made some assumptions that only hold for integer solutions"
    assert is_integer(sols_pred), "we have made some assumptions that only hold for integer solutions"

    cost_dist_exp_true, cost_dist_exp_true_norm = norm(expectation(cost_dists_true))
    cost_dist_var_true = variance(cost_dists_true) / cost_dist_exp_true_norm
    cost_dist_exp_pred, cost_dist_exp_pred_norm = norm(expectation(cost_dists_pred))
    cost_dist_var_pred = variance(cost_dists_pred) / cost_dist_exp_pred_norm

    # wait-and-see solution value with true cost realizations
    obj_true = torch.bmm(costs_true.unsqueeze(1), sols_true.unsqueeze(2)).squeeze(2)
    # predicted solution value with learned cost expectations
    obj_pred = torch.bmm(costs_pred.unsqueeze(1), sols_pred.unsqueeze(2)).squeeze(2)
    # predicted solution value with true cost realizations
    obj_realized = torch.bmm(costs_true.unsqueeze(1), sols_pred.unsqueeze(2)).squeeze(2)
    # predicted solution value with true cost expectations
    obj_expected = torch.bmm(cost_dist_exp_true.unsqueeze(1), sols_pred.unsqueeze(2)).squeeze(2)

    regret = obj_realized - obj_true
    # disappointment is positive when the realized objective is greater than the predicted one
    disappointment = (obj_realized - obj_expected).clamp(min=0)
    # surprise is positive when the realized objective is less than the predicted one (the predicted one is greater)
    surprise = (obj_expected - obj_realized).clamp(min=0)
    # these definitions flip when maximizing
    if model_sense == pyepo.EPO.MAXIMIZE:
        regret *= -1
        disappointment, surprise = surprise, disappointment

    cost_err = F.mse_loss(costs_pred, costs_true, reduction="mean")
    if class_weights is not None:
        sample_weights = torch.empty_like(sols_true)
        sample_weights[sols_true == 0] = class_weights[0]
        sample_weights[sols_true == 1] = class_weights[1]
    else:
        sample_weights = None
    decision_err = F.binary_cross_entropy(sols_pred, sols_true, weight=sample_weights, reduction="mean")

    # we compare cost distributions by their (normalized) expectation and variance
    cost_dist_exp_err = F.mse_loss(cost_dist_exp_pred, cost_dist_exp_true, reduction="mean")
    cost_dist_var_err = F.mse_loss(cost_dist_var_pred, cost_dist_var_true, reduction="mean")

    return EvalMetrics(
        cost_err,
        decision_err,
        regret.mean(),
        disappointment.mean(),
        surprise.mean(),
        obj_true,
        obj_pred,
        obj_expected,
        obj_realized,
        cost_dist_exp_err,
        cost_dist_var_err,
    )
