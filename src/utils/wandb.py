import argparse
import torch
import wandb

from collections.abc import Sequence
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


def get_wandb_name(args: argparse.Namespace, argparser: argparse.ArgumentParser):
    nondefault_values = []
    for name, value in vars(args).items():
        default_value = argparser.get_default(name)
        if value != default_value and "wandb" not in name:
            nondefault_values.append((name, value))

    if len(nondefault_values) == 0:
        return None

    name = "_".join(f"{name}:{value}" for name, value in nondefault_values)
    return name


class WandBHistogram(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        super().__init__(output_transform, device)
        self.reset()

    def reset(self):
        self.samples = []

    def update(self, sample):
        if isinstance(sample, Sequence):
            raise TypeError("received a sequence but expected a single value")
        self.samples.append(sample)

    def compute(self):
        if len(self.samples) == 0:
            raise NotComputableError("no samples")

        return wandb.Histogram(self.samples)
