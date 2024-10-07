import argparse
import json
import os
import pathlib
import torch
import wandb

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric

from typing import Dict


def get_friendly_name(args: argparse.Namespace, argparser: argparse.ArgumentParser):
    nondefault_values = []
    for name, value in vars(args).items():
        default_value = argparser.get_default(name)
        if value != default_value and "wandb" not in name:
            nondefault_values.append((name, value))

    if len(nondefault_values) == 0:
        return None

    name = "_".join(f"{name}:{value}" for name, value in nondefault_values)
    return name


def record_metrics(metrics: Dict[str, Metric], epoch: int):
    for name, metric in metrics.items():
        try:
            val = metric.compute()
            if isinstance(val, torch.Tensor) and val.dim() > 0:
                wandb.log({name: wandb.Histogram(val)}, step=epoch)
                wandb.log({name + "_mean": val.mean()}, step=epoch)
            else:
                wandb.log({name: val}, step=epoch)
        except NotComputableError:
            pass


def save_metrics(metrics: Dict[str, Metric], epoch: int):
    artifact_name = f"metrics_{wandb.run.id}"
    artifact = wandb.Artifact(artifact_name, "metrics")
    # artifact.aliases.append(f"epoch={epoch}")
    tmp_dir = pathlib.Path(os.environ.get("SLURM_TMPDIR", "/tmp"))

    for name, metric in metrics.items():
        name = name.replace(os.path.sep, "_")
        val = metric.compute()
        path = tmp_dir / f"{name}.pt"
        torch.save(torch.as_tensor(val), path)
        artifact.add_file(path)

    wandb.log_artifact(artifact)


def save_model(model: torch.nn.Module, epoch: int):
    artifact_name = f"model_{wandb.run.id}"
    artifact = wandb.Artifact(artifact_name, "model")
    # artifact.aliases.append(f"epoch={epoch}")
    tmp_dir = pathlib.Path(os.environ.get("SLURM_TMPDIR", "/tmp"))

    model_path = tmp_dir / "weights.pt"
    torch.save(model.state_dict(), model_path)
    artifact.add_file(model_path)

    config_path = tmp_dir / "config.json"
    json.dump(wandb.run.config.as_dict(), open(config_path, "w"))
    artifact.add_file(config_path)

    wandb.log_artifact(artifact)


class WandBHistogram(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        super().__init__(output_transform, device)
        self.reset()

    def reset(self):
        self.samples = []

    def update(self, samples):
        self.samples.append(samples.cpu())

    def compute(self):
        if len(self.samples) == 0:
            raise NotComputableError("no samples")

        if isinstance(self.samples[0], torch.Tensor):
            samples = torch.cat(self.samples, dim=-1)
        else:
            samples = self.samples
        return wandb.Histogram(samples)
