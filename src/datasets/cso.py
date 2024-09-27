import numpy as np
import torch

from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from distributions.twopoint import TwoPoint
from utils.utils import is_integer, norm, norm_normal


class CSLPDataset(optDataset):
    def __init__(
        self,
        model: optModel,
        feats: torch.Tensor,
        costs: torch.Tensor,
        cost_dist: str,
        cost_dist_params: torch.Tensor,
    ):
        # super sets self.feats, self.costs and calculates self.sols, self.objs according to model
        super().__init__(model, feats, costs)

        assert cost_dist in ["normal", "twopoint"], f"unsupported distribution {cost_dist}"
        self.cost_dist = cost_dist
        self.cost_dist_params = cost_dist_params

        self.is_integer = is_integer(self.sols)
        if self.is_integer:
            self.class_weights = compute_class_weight(
                class_weight="balanced", classes=np.array([0, 1]), y=self.sols.flatten().numpy()
            )

        self.feat_scaler = StandardScaler().fit(self.feats)

    def __getitem__(self, index):
        return self.feats[index], self.costs[index], self.sols[index], self.objs[index], self.cost_dist_params[index]

    def collate_batch(self, batch):
        feats, costs, sols, objs, cost_dist_params = map(torch.cat, zip(*batch))
        if self.cost_dist == "twopoint":
            cost_dist_lows, cost_dist_highs, cost_dist_probs = torch.chunk(cost_dist_params, 3, dim=-1)
            cost_dists = TwoPoint(cost_dist_lows, cost_dist_highs, cost_dist_probs)
        elif self.cost_dist == "normal":
            cost_dist_mean, cost_dist_std = torch.chunk(cost_dist_params, 2, dim=-1)
            cost_dists = torch.distributions.Normal(cost_dist_mean, cost_dist_std)
        else:
            raise ValueError(f"unknown distribution {self.cost_dist}")
        return feats, costs, sols, objs, cost_dists

    def norm_batch(self, batch):
        feats, costs, sols, objs, cost_dist_params = batch

        feats_normed = torch.as_tensor(self.feat_scaler.transform(feats))

        costs_normed = norm(costs)

        assert self.is_integer, "not sure how to normalize non-integer solutions"
        sols_normed = sols

        objs_normed = torch.bmm(costs_normed, sols)

        if self.cost_dist == "normal":
            cost_dist_mean, cost_dist_std = torch.chunk(cost_dist_params, 2)
            cost_dist = torch.distributions.Normal(cost_dist_mean, cost_dist_std)
            cost_dist_normed = norm_normal(cost_dist)
        else:
            raise ValueError(f"not sure how to normalize distribution {self.cost_dist}")

        return feats_normed, costs_normed, sols_normed, objs_normed, cost_dist_normed
