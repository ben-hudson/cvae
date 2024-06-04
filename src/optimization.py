import torch
import torch.nn.functional as F

from cooper import ConstrainedMinimizationProblem as CMP, CMPState
from torch.distributions import kl_divergence

class InverseLinearOptimization(CMP):
    def __init__(self, kld_weight: int, device: str='cpu'):
        super().__init__(is_constrained=True)

        assert kld_weight >= 0 and kld_weight <= 1, f'kld weight must be between 0 and 1, it is {kld_weight}'
        self.kld_weight = kld_weight
        self.device = device

    def closure(self, model, batch):
        x, y, W, h, q, Q = batch

        x = x.float().to(self.device)
        y = y.float().to(self.device)
        Q = Q.float().to(self.device)

        priors, posteriors, sample = model(y, x)

        y_sample = sample.y.unsqueeze(2) # column vector
        q_sample = sample.y.unsqueeze(1) # row vector
        W_sample = sample.W
        h_sample = sample.h.unsqueeze(2)

        # "absolute objective error" loss from https://arxiv.org/abs/2006.08923
        Q_sample = torch.bmm(q_sample, y_sample).squeeze()
        aoe = F.l1_loss(Q_sample, Q).sum()

        # constraint is Wy = h --> Wy - h = 0
        defect = (torch.bmm(W_sample, y_sample) - h_sample).sum()

        # want posterior to be close to the prior
        kld = torch.zeros(1, device=self.device)
        for p, q in zip(priors, posteriors):
            kld += kl_divergence(p, q).sum()

        loss = self.kld_weight*kld + (1 - self.kld_weight)*aoe
        return CMPState(loss=loss, eq_defect=defect, misc={'kld': kld, 'aoe': aoe})
