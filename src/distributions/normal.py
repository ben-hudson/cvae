import torch
import torch.distributions as D


class Normal(D.Normal):
    def _expectation(self):
        return self.loc

    def _variance(self):
        return self.scale**2

    def to(self, device: torch.device):
        return self.__class__(self.loc.to(device), self.scale.to(device))
