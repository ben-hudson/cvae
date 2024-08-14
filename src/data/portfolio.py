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
                scale[torch.isclose(scale, torch.zeros_like(scale))] = 1.0

            for field, mean in self.means._asdict().items():
                assert not mean.requires_grad, f"{field} mean requires grad"

            for field, scale in self.scales._asdict().items():
                assert not scale.requires_grad, f"{field} scale requires grad"

            self.feats_normed, self.costs_normed, self.sols_normed, self.objs_normed = self.norm(
                self.feats, self.costs, self.sols, self.objs
            )

    def norm(self, feats=None, costs=None, sols=None, objs=None):
        assert self.is_normed, "can't call norm on an unnormed dataset"
        feats_normed = (
            (feats - self.means.feats.to(feats.device)) / self.scales.feats.to(feats.device)
            if feats is not None
            else None
        )
        costs_normed = (
            (costs - self.means.costs.to(costs.device)) / self.scales.costs.to(costs.device)
            if costs is not None
            else None
        )
        sols_normed = (
            (sols - self.means.sol.to(sols.device)) / self.scales.sol.to(sols.device) if sols is not None else None
        )
        objs_normed = (
            (objs - self.means.obj.to(objs.device)) / self.scales.obj.to(objs.device) if objs is not None else None
        )

        return PortfolioDatapoint(feats_normed, costs_normed, sols_normed, objs_normed)

    def unnorm(self, feats_normed=None, costs_normed=None, sols_normed=None, objs_normed=None):
        assert self.is_normed, "can't call unnorm on an unnormed dataset"
        feats = (
            self.means.feats.to(feats_normed.device) + self.scales.feats.to(feats_normed.device) * feats_normed
            if feats_normed is not None
            else None
        )
        costs = (
            self.means.costs.to(costs_normed.device) + self.scales.costs.to(costs_normed.device) * costs_normed
            if costs_normed is not None
            else None
        )
        sols = (
            self.means.sol.to(sols_normed.device) + self.scales.sol.to(sols_normed.device) * sols_normed
            if sols_normed is not None
            else None
        )
        objs = (
            self.means.obj.to(objs_normed.device) + self.scales.obj.to(objs_normed.device) * objs_normed
            if objs_normed is not None
            else None
        )

        return PortfolioDatapoint(feats, costs, sols, objs)

    def __getitem__(self, index):
        if self.is_normed:
            return PortfolioDatapoint(
                self.feats_normed[index], self.costs_normed[index], self.sols_normed[index], self.objs_normed[index]
            )
        else:
            return PortfolioDatapoint(self.feats[index], self.costs[index], self.sols[index], self.objs[index])

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
