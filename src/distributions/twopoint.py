import torch
import torch.distributions as D


class TwoPoint(D.TransformedDistribution):
    def __init__(self, low, high, prob):
        self.lo = torch.as_tensor(low)
        self.hi = torch.as_tensor(high)
        self.prob = torch.as_tensor(prob)

        # a two-point distribution is just a shifted and scaled bernoulli distribtion
        base_distribution = D.Bernoulli(self.prob)
        transform = D.transforms.AffineTransform(self.lo, self.hi - self.lo)
        super().__init__(base_distribution, [transform])

    def _expectation(self):
        # the expectation a distribution is E[X]
        return self.hi * self.prob + self.lo * (1 - self.prob)

    def _variance(self):
        # the variance a distribution is Var(X) = E[X^2] - E[X]^2
        return self.hi**2 * self.prob + self.lo**2 * (1 - self.prob) - self._expectation() ** 2

    def to(self, device: torch.device):
        return self.__class__(self.lo.to(device), self.hi.to(device), self.prob.to(device))

    # hacky version because the built-in one doesn't work
    def log_prob(self, value):
        probs = torch.zeros_like(self.prob)
        probs[value == self.lo] = 1 - self.prob[value == self.lo]
        probs[value == self.hi] = self.prob[value == self.hi]
        return probs.log()
