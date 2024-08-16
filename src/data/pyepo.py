import pyepo.data
import pyepo.model
import torch
import pyepo

from collections import namedtuple

PyEPODatapoint = namedtuple("PyEPODatapoint", "feats costs sols objs")


class PyEPODataset(pyepo.data.dataset.optDataset):
    def __init__(self, model, feats, costs, norm=True):
        feats = torch.FloatTensor(feats)
        costs = torch.FloatTensor(costs)
        super().__init__(model, feats, costs)
        # super sets self.feats and self.costs
        self.sols = torch.FloatTensor(self.sols)
        self.objs = torch.FloatTensor(self.objs)

        self.is_integer = ((self.sols == 0) | (self.sols == 1)).all()

        self.is_normed = norm
        if self.is_normed:
            self.means = PyEPODatapoint(
                self.feats.mean(dim=0, keepdim=True),
                self.costs.mean(dim=0, keepdim=True),
                self.sols.mean(dim=0, keepdim=True),
                self.objs.mean(dim=0, keepdim=True),
            )

            self.scales = PyEPODatapoint(
                self.feats.std(dim=0, correction=0, keepdim=True),
                self.costs.std(dim=0, correction=0, keepdim=True),
                self.sols.std(dim=0, correction=0, keepdim=True),
                self.objs.std(dim=0, correction=0, keepdim=True),
            )

            # if solutions are binary, we don't normalize them
            if self.is_integer:
                self.means.sols[:] = 0.0
                self.scales.sols[:] = 1.0

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
            (sols - self.means.sols.to(sols.device)) / self.scales.sols.to(sols.device) if sols is not None else None
        )
        objs_normed = (
            (objs - self.means.objs.to(objs.device)) / self.scales.objs.to(objs.device) if objs is not None else None
        )

        return PyEPODatapoint(feats_normed, costs_normed, sols_normed, objs_normed)

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
            self.means.sols.to(sols_normed.device) + self.scales.sols.to(sols_normed.device) * sols_normed
            if sols_normed is not None
            else None
        )
        objs = (
            self.means.objs.to(objs_normed.device) + self.scales.objs.to(objs_normed.device) * objs_normed
            if objs_normed is not None
            else None
        )

        return PyEPODatapoint(feats, costs, sols, objs)

    def __getitem__(self, index):
        if self.is_normed:
            return PyEPODatapoint(
                self.feats_normed[index], self.costs_normed[index], self.sols_normed[index], self.objs_normed[index]
            )
        else:
            return PyEPODatapoint(self.feats[index], self.costs[index], self.sols[index], self.objs[index])

    @property
    def covariance(self):
        return self.model.covariance

    @property
    def risk_level(self):
        return self.model.risk_level
