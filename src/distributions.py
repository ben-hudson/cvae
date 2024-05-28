import torch.distributions as dist
import torch.nn.functional as F

class ReparametrizedBernoulli(dist.Bernoulli):
    def rsample(self, tau=1):
        return F.gumbel_softmax(self.logits, tau=tau, hard=True)
