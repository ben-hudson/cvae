import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class Accumulator(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        super().__init__(output_transform, device)
        self.reset()

    def reset(self):
        self.samples = []

    def update(self, samples):
        self.samples.append(samples.squeeze().to(self._device))

    def compute(self):
        if len(self.samples) == 0:
            raise NotComputableError("no samples")

        if isinstance(self.samples[0], torch.Tensor):
            samples = torch.cat(self.samples, dim=-1)
        else:
            samples = self.samples
        return samples
