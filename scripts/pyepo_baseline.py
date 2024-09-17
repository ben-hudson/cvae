import argparse
import numpy as np
import pyepo
import pyepo.metric
import torch
import torch.utils
import tqdm

from collections import defaultdict
from ignite.metrics import Average
from data.pyepo import PyEPODataset
from models.parallel_solver import ParallelSolver
from models.risk_averse import CVaRShortestPath
from utils import get_val_metrics, get_wandb_name, WandBHistogram


def get_argparser():
    parser = argparse.ArgumentParser("Run a baseline on a dataset")

    dataset_args = parser.add_argument_group("dataset", description="Dataset arguments")
    dataset_args.add_argument("dataset", type=str, choices=["shortestpath", "portfolio"], help="Dataset to generate")
    dataset_args.add_argument("--n_samples", type=int, default=2000, help="Number of samples to generate")
    dataset_args.add_argument("--n_features", type=int, default=5, help="Number of features")
    dataset_args.add_argument("--degree", type=int, default=1, help="Polynomial degree for encoding function")
    dataset_args.add_argument("--noise_width", type=float, default=0.5, help="Half-width of latent uniform noise")
    dataset_args.add_argument("--workers", type=int, default=2, help="Number of DataLoader workers")
    dataset_args.add_argument("--seed", type=int, default=135, help="RNG seed")

    model_args = parser.add_argument_group("model", description="Model arguments")
    model_args.add_argument(
        "baseline", type=str, choices=["oracle", "random", "mean", "ra_oracle"], help="Baseline to evaluate"
    )

    train_args = parser.add_argument_group("training", description="Training arguments")
    train_args.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")

    model_args = parser.add_argument_group("logging", description="Logging arguments")
    model_args.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    model_args.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags")

    return parser


def report_metrics(metrics: dict, step: int, use_wandb: bool):
    log = {name: avg.compute() for name, avg in metrics.items()}

    if use_wandb:
        wandb.log(log, step=step)
    else:
        print(f"step: {step}")
        for name, val in log.items():
            try:
                print(f"{name}: {val:.4f}")
            except Exception as e:
                print(f"{name}: {e}")


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()

    args.use_wandb = args.wandb_project is not None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            config=args,
            name=get_wandb_name(args, argparser),
            tags=["baseline"] + args.wandb_tags,
        )

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # to generate the "oracle" predictions we generate the data with no noise, and then add noise back in to get the observations
    assert args.noise_width <= 1, "noise width must be at most 1"
    if args.dataset == "shortestpath":
        is_integer = True
        grid = (5, 5)
        feats, costs_expected = pyepo.data.shortestpath.genData(
            args.n_samples, args.n_features, grid, deg=args.degree, noise_width=0, seed=args.seed
        )

        feats = torch.FloatTensor(feats)
        costs_expected = torch.FloatTensor(costs_expected)

        costs_std = args.noise_width / costs_expected.abs()

        cost_dist = "normal"
        cost_dist_params = torch.cat([costs_expected, costs_std], dim=-1)
        costs = torch.distributions.Normal(costs_expected, costs_std).sample()

        data_model = pyepo.model.grb.shortestPathModel(grid)

    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    # we want to report the running average every so often
    # so we divide the total number of samples into "batches"
    batch_size = args.n_samples // args.max_epochs

    dataset = PyEPODataset(data_model, feats, costs, cost_dist_params)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers, drop_last=False
    )

    cost_marginal_mean = costs.mean(dim=0)
    cost_marginal_std = costs.std(dim=0, correction=0)

    if args.baseline != "ra_oracle":
        solver = ParallelSolver(args.workers, pyepo.model.grb.shortestPathModel, grid)
    else:
        solver = ParallelSolver(args.workers, CVaRShortestPath, grid, 0.99, tail="right")

    metrics = defaultdict(Average)
    # override these
    metrics["val/obj_true"] = WandBHistogram()
    metrics["val/obj_pred"] = WandBHistogram()
    metrics["val/obj_expected"] = WandBHistogram()
    metrics["val/obj_realized"] = WandBHistogram()

    for epoch, batch in enumerate(dataloader):
        feats, costs, sols, objs, cost_dist_params = batch
        cost_dist_mean, cost_dist_std = cost_dist_params.chunk(2, dim=-1)
        cost_dist = torch.distributions.Normal(cost_dist_mean, cost_dist_std)

        if args.baseline == "ra_oracle":
            y_pred = cost_dist_params
            costs_pred = cost_dist_mean
        elif args.baseline == "oracle":
            y_pred = costs_pred = cost_dist_mean
        elif args.baseline == "random":
            y_pred = costs_pred = cost_marginal_mean + cost_marginal_std * torch.randn_like(costs)
        elif args.baseline == "mean":
            y_pred = costs_pred = cost_marginal_mean.expand_as(costs)

        sols_pred = solver(y_pred)

        objs_pred = torch.bmm(costs_pred.unsqueeze(1), sols_pred.unsqueeze(2)).squeeze(2)

        eval_metrics = get_val_metrics(
            costs,
            sols,
            objs,
            cost_dist,
            costs_pred,
            sols_pred,
            objs_pred,
            cost_dist,
            data_model.modelSense,
            dataset.is_integer,
            dataset.class_weights,
        )

        for name, value in eval_metrics._asdict().items():
            metrics["val/" + name].update(value)

        report_metrics(metrics, epoch, args.use_wandb)

    report_metrics(metrics, args.max_epochs, args.use_wandb)
