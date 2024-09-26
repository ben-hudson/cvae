import torch
import torch.distributions as D


class TwoPoint(D.TransformedDistribution):
    def __init__(self, lows, highs, probs):
        self.lows = torch.as_tensor(lows)
        self.highs = torch.as_tensor(highs)
        self.probs = torch.as_tensor(probs)

        # a two-point distribution is just a shifted and scaled bernoulli distribtion
        base_distribution = D.Bernoulli(self.probs)
        transform = D.transforms.AffineTransform(self.lows, self.highs - self.lows)
        super().__init__(base_distribution, [transform])

    def _expectation(self):
        # the expectation a distribution is E[X]
        return self.highs * self.probs + self.lows * (1 - self.probs)

    def _variance(self):
        # the variance a distribution is Var(X) = E[X^2] - E[X]^2
        return self.highs**2 * self.probs + self.lows**2 * (1 - self.probs) - self.loc
