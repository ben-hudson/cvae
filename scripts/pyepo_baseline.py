import argparse
import numpy as np
import pyepo
import pyepo.metric
import torch
import tqdm

from collections import defaultdict
from ignite.metrics import Average
from models.risk_averse import CVaRShortestPath
from utils import get_sample_val_metrics, get_wandb_name, WandBHistogram


def get_argparser():
    parser = argparse.ArgumentParser("Run a baseline on a dataset")

    dataset_args = parser.add_argument_group("dataset", description="Dataset arguments")
    dataset_args.add_argument("dataset", type=str, choices=["shortestpath", "portfolio"], help="Dataset to generate")
    dataset_args.add_argument("--n_samples", type=int, default=2000, help="Number of samples to generate")
    dataset_args.add_argument("--n_features", type=int, default=5, help="Number of features")
    dataset_args.add_argument("--degree", type=int, default=1, help="Polynomial degree for encoding function")
    dataset_args.add_argument("--noise_width", type=float, default=0.5, help="Half-width of latent uniform noise")
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
        costs = torch.distributions.Normal(costs_expected, costs_std).sample()

        data_model = pyepo.model.grb.shortestPathModel(grid)

    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    cost_marginal_mean = costs.mean(dim=0)
    cost_marginal_std = costs.std(dim=0, correction=0)

    metrics = defaultdict(Average)
    # override these two
    metrics["val/obj_true"] = WandBHistogram()
    metrics["val/obj_realized"] = WandBHistogram()

    # we want to report the running average every so often
    # so we divide the total number of samples into "batches"
    batch_size = args.n_samples // args.max_epochs

    objs_true = []
    sols_true = []

    for i in tqdm.trange(args.n_samples):
        cost_true = costs[i]

        if cost_dist == "uniform":
            cost_dist_true = torch.distributions.Uniform(cost_dist_los[i], cost_dist_his[i])
        elif cost_dist == "normal":
            cost_dist_true = torch.distributions.Normal(costs_expected[i], costs_std[i])
        else:
            raise ValueError(f"unknown distribution {cost_dist}")

        data_model.setObj(cost_true)
        sol_true, obj_true = data_model.solve()
        sol_true = torch.FloatTensor(sol_true)
        sols_true.append(sol_true)
        objs_true.append(obj_true)

        if args.baseline == "ra_oracle":
            cost_pred = costs_expected[i]
            cost_pred_std = costs_std[i]
            ra_data_model = CVaRShortestPath(grid, cost_pred, cost_pred_std, 0.99, tail="right")
            sol_pred, _ = ra_data_model.solve()

        else:
            if args.baseline == "oracle":
                cost_pred = costs_expected[i]
            elif args.baseline == "random":
                cost_pred = cost_marginal_mean + cost_marginal_std * torch.randn_like(cost_true)
            elif args.baseline == "mean":
                cost_pred = cost_marginal_mean
            else:
                raise ValueError(f"unknown baseline {args.model}")

            data_model.setObj(cost_pred)
            sol_pred, _ = data_model.solve()

        cost_dist_pred = torch.distributions.Normal(costs_expected[i], costs_std[i])

        sol_pred = (torch.FloatTensor(sol_pred) > 0.5).to(torch.float32)
        obj_pred = torch.dot(cost_pred, sol_pred)

        sample_metrics = get_sample_val_metrics(
            data_model,
            cost_true,
            sol_true,
            obj_true,
            cost_dist_true,
            cost_pred,
            sol_pred,
            obj_pred,
            cost_dist_pred,
            is_integer,
        )

        for name, value in sample_metrics._asdict().items():
            if not np.isinf(value):
                metrics["val/" + name].update(value)

        if i % batch_size == 0:
            report_metrics(metrics, i // batch_size, args.use_wandb)

    report_metrics(metrics, args.max_epochs, args.use_wandb)
