import pyepo.data
import pyepo.model
import torch
import pyepo

from collections import namedtuple

PortfolioDatapoint = namedtuple("PortfolioDatapoint", "feats costs sol obj")

class PortfolioDataset(pyepo.data.dataset.optDataset):
    def __init__(self, model, feats, costs, norm=True):
        feats = torch.FloatTensor(feats)
        costs = torch.FloatTensor(costs)

        super().__init__(model, feats, costs)
        self.sols = torch.FloatTensor(self.sols)
        self.objs = torch.FloatTensor(self.objs)

        self.is_normed = norm
        if self.is_normed:
            self.means = PortfolioDatapoint(
                self.feats.mean(dim=0, keepdim=True),
                self.costs.mean(dim=0, keepdim=True),
                self.sols.mean(dim=0, keepdim=True),
                self.objs.mean(dim=0, keepdim=True),
            )

            self.scales = PortfolioDatapoint(
                self.feats.std(dim=0, correction=0, keepdim=True),
                self.costs.std(dim=0, correction=0, keepdim=True),
                self.sols.std(dim=0, correction=0, keepdim=True),
                self.objs.std(dim=0, correction=0, keepdim=True),
            )

            # prevent division by 0, just like sklearn.preprocessing.StandardScaler
            for scale in self.scales:
                scale[torch.isclose(scale, torch.zeros_like(scale))] = 1.

            for field, mean in self.means._asdict().items():
                assert not mean.requires_grad, f"{field} mean requires grad"

            for field, scale in self.scales._asdict().items():
                assert not scale.requires_grad, f"{field} scale requires grad"

            self.feats_normed, self.costs_normed, self.sols_normed, self.objs_normed = self.norm(self.feats, self.costs, self.sols, self.objs)

    def norm(self, feats, costs, sols, objs):
        assert self.is_normed, "can't call norm on an unnormed dataset"
        feats_normed = (feats - self.means.feats.to(feats.device))/self.scales.feats.to(feats.device)
        costs_normed = (costs- self.means.costs.to(costs.device))/self.scales.costs.to(costs.device)
        sols_normed = (sols - self.means.sol.to(sols.device))/self.scales.sol.to(sols.device)
        objs_normed = (objs - self.means.obj.to(objs.device))/self.scales.obj.to(objs.device)

        return PortfolioDatapoint(feats_normed, costs_normed, sols_normed, objs_normed)

    def unnorm(self, feats_normed, costs_normed, sols_normed, objs_normed):
        assert self.is_normed, "can't call unnorm on an unnormed dataset"
        feats = self.means.feats.to(feats_normed.device) + self.scales.feats.to(feats_normed.device)*feats_normed
        costs = self.means.costs.to(costs_normed.device) + self.scales.costs.to(costs_normed.device)*costs_normed
        sols = self.means.sol.to(sols_normed.device) + self.scales.sol.to(sols_normed.device)*sols_normed
        objs = self.means.obj.to(objs_normed.device) + self.scales.obj.to(objs_normed.device)*objs_normed

        return PortfolioDatapoint(feats, costs, sols, objs)

    def __getitem__(self, index):
        if self.is_normed:
            return PortfolioDatapoint(
                self.feats_normed[index],
                self.costs_normed[index],
                self.sols_normed[index],
                self.objs_normed[index]
            )
        else:
            return PortfolioDatapoint(
                self.feats[index],
                self.costs[index],
                self.sols[index],
                self.objs[index]
            )

    @property
    def covariance(self):
        return self.model.covariance

    @property
    def risk_level(self):
        return self.model.risk_level

if __name__ == "__main__":
    n_assets = 50
    cov, feats, costs = pyepo.data.portfolio.genData(1000, 5, n_assets, deg=1, noise_level=0.5, seed=135)
    model = pyepo.model.grb.portfolioModel(n_assets, cov, 2.25)
    dataset = PortfolioDataset(model, feats, costs, norm=True)