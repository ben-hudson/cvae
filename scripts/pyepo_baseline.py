import argparse
import numpy as np
import pyepo
import pyepo.metric
import torch
import tqdm

from ignite.metrics import Average
from metrics.util import get_val_metrics_sample


def get_argparser():
    parser = argparse.ArgumentParser("Train an MLP to approximate an LP solver using constrained optimization")

    dataset_args = parser.add_argument_group("dataset", description="Dataset arguments")
    dataset_args.add_argument("dataset", type=str, choices=["shortestpath", "portfolio"], help="Dataset to generate")
    dataset_args.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate")
    dataset_args.add_argument("--n_features", type=int, default=5, help="Number of features")
    dataset_args.add_argument("--degree", type=int, default=1, help="Polynomial degree for encoding function")
    dataset_args.add_argument("--noise_width", type=float, default=0.5, help="Half-width of latent uniform noise")
    dataset_args.add_argument("--seed", type=int, default=135, help="RNG seed")

    model_args = parser.add_argument_group("model", description="Model arguments")
    model_args.add_argument("baseline", type=str, choices=["oracle", "random", "mean"], help="Baseline to evaluate")

    train_args = parser.add_argument_group("training", description="Training arguments")
    train_args.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")

    model_args = parser.add_argument_group("logging", description="Logging arguments")
    model_args.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    model_args.add_argument("--wandb_exp", type=str, default=None, help="WandB experiment name")
    model_args.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags")

    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    args.use_wandb = args.wandb_project is not None

    assert args.noise_width <= 1, "noise width must be at most 1 to prevent negative values"

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            config=args,
        )

    # to generate the "oracle" predictions we generate the data with no noise, and then add noise back in to get the dataset
    if args.dataset == "shortestpath":
        is_integer = True
        grid = (5, 5)
        data_model = pyepo.model.grb.shortestPathModel(grid)
        feats, expected_costs = pyepo.data.shortestpath.genData(
            args.n_samples, args.n_features, grid, deg=args.degree, noise_width=0, seed=args.seed
        )
        # the shortest path problem has multiplicative uniform noise
        epsilon = rng.uniform(1 - args.noise_width, 1 + args.noise_width, size=expected_costs.shape)
        costs = expected_costs * epsilon

    else:
        raise ValueError("NYI")

    expected_costs = torch.FloatTensor(expected_costs)
    costs = torch.FloatTensor(costs)

    cost_mean = costs.mean(dim=0)
    cost_std = costs.std(dim=0, correction=0)

    metric_names = [
        "val/cost_err",
        "val/decision_err",
        "val/regret",
        "val/spo_loss",
        "val/abs_obj",
    ]
    metrics = {}
    for name in metric_names:
        metrics[name] = Average()

    for i in tqdm.trange(len(costs)):
        cost_true = costs[i]

        data_model.setObj(cost_true)
        sol_true, obj_true = data_model.solve()
        sol_true = torch.FloatTensor(sol_true)

        if args.baseline == "oracle":
            cost_pred = expected_costs[i]
        elif args.baseline == "random":
            cost_pred = cost_mean + cost_std * torch.randn_like(cost_true)
        elif args.baseline == "mean":
            cost_pred = cost_mean
        else:
            raise ValueError(f"unknown baseline {args.model}")

        data_model.setObj(cost_pred)
        sol_pred, obj_pred = data_model.solve()
        sol_pred = torch.FloatTensor(sol_pred)

        sample_metrics = get_val_metrics_sample(
            data_model, cost_true, cost_pred, sol_true, sol_pred, obj_true, obj_pred, is_integer
        )

        metrics["val/cost_err"].update(sample_metrics.cost_err)
        metrics["val/decision_err"].update(sample_metrics.decision_err)
        metrics["val/regret"].update(sample_metrics.regret)
        metrics["val/spo_loss"].update(sample_metrics.spo_loss)
        metrics["val/abs_obj"].update(sample_metrics.abs_obj)

    log = {name: avg.compute() for name, avg in metrics.items()}
    log["val/pyepo_regret_norm"] = log["val/spo_loss"] / (log["val/abs_obj"] + 1e-7)

    if args.use_wandb:
        for epoch in range(args.max_epochs):
            wandb.log(log, step=epoch)

    else:
        for name, val in log.items():
            print(f"{name}: {val:.4f}")
