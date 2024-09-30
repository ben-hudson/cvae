import torch
import torch.distributions as D


class TwoPoint(D.TransformedDistribution):
    def __init__(self, lows, highs, probs):
        self.los = torch.as_tensor(lows)
        self.his = torch.as_tensor(highs)
        self.probs = torch.as_tensor(probs)

        # a two-point distribution is just a shifted and scaled bernoulli distribtion
        base_distribution = D.Bernoulli(self.probs)
        transform = D.transforms.AffineTransform(self.los, self.his - self.los)
        super().__init__(base_distribution, [transform])

    def _expectation(self):
        # the expectation a distribution is E[X]
        return self.his * self.probs + self.los * (1 - self.probs)

    def _variance(self):
        # the variance a distribution is Var(X) = E[X^2] - E[X]^2
        return self.his**2 * self.probs + self.los**2 * (1 - self.probs) - self._expectation() ** 2

    def to(self, device: torch.device):
        return self.__class__(self.los.to(device), self.his.to(device), self.probs.to(device))
