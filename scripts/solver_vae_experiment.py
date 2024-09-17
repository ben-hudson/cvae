import argparse
import os
import pathlib
import pyepo
import pyepo.data
import pyepo.metric
import pyepo.model
import torch
import torch.distributions as D
import torch.nn.functional as F
import tqdm

from collections import defaultdict
from data.pyepo import PyEPODataset
from ignite.exceptions import NotComputableError
from ignite.metrics import Average
from models.risk_averse import VaRShortestPath, CVaRShortestPath
from models.solver_vae import SolverVAE
from models.parallel_solver import ParallelSolver
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Sequence, Dict, Tuple
from utils import get_val_metrics, get_wandb_name, WandBHistogram


def get_argparser():
    parser = argparse.ArgumentParser("Train an MLP to approximate an LP solver using constrained optimization")

    dataset_args = parser.add_argument_group("dataset", description="Dataset arguments")
    dataset_args.add_argument("dataset", type=str, choices=["shortestpath", "portfolio"], help="Dataset to generate")
    dataset_args.add_argument("--n_samples", type=int, default=2000, help="Number of samples to generate")
    dataset_args.add_argument("--n_features", type=int, default=5, help="Number of features")
    dataset_args.add_argument("--degree", type=int, default=1, help="Polynomial degree for encoding function")
    dataset_args.add_argument("--noise_width", type=float, default=0.5, help="Half-width of latent uniform noise")
    dataset_args.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    dataset_args.add_argument("--workers", type=int, default=2, help="Number of DataLoader workers")
    dataset_args.add_argument("--seed", type=int, default=135, help="RNG seed")

    model_args = parser.add_argument_group("model", description="Model arguments")
    model_args.add_argument("--mlp_hidden_dim", type=int, default=64, help="Dimension of hidden layers in MLPs")
    model_args.add_argument("--mlp_hidden_layers", type=int, default=2, help="Number of hidden layers in MLPs")
    model_args.add_argument("--latent_dist", type=str, default="normal", choices=["normal"], help="Latent distribution")
    model_args.add_argument("--latent_dim", type=int, default=10, help="Latent dimension")

    train_args = parser.add_argument_group("training", description="Training arguments")
    train_args.add_argument("--no_gpu", action="store_true", help="Do not use the GPU even if one is available")
    train_args.add_argument("--lr", type=float, default=1e-5, help="Optimizer learning rate")
    train_args.add_argument(
        "--kld_weight", type=float, default=0.5, help="Relative weighting of KLD and reconstruction loss"
    )
    train_args.add_argument("--momentum", type=float, default=8e-2, help="Optimizer momentum")
    train_args.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")
    train_args.add_argument("--max_hours", type=int, default=3, help="Maximum hours to train")
    train_args.add_argument("--save_every", type=int, default=10, help="Save model weights every n epochs")

    eval_args = parser.add_argument_group("evaluation", description="Evaluation arguments")
    eval_args.add_argument("--eval_every", type=int, default=10, help="Evaluate every n epochs")
    eval_args.add_argument(
        "--chance_constraint_budget", type=float, default=None, help="Chance constraint cost threshold"
    )
    eval_args.add_argument(
        "--risk_level", type=float, default=0.5, help="Risk level (probability) for risk-averse decision-making"
    )

    model_args = parser.add_argument_group("logging", description="Logging arguments")
    model_args.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    model_args.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags")

    return parser


def get_train_metrics(
    prior: D.Distribution,
    posterior: D.Distribution,
    sols: torch.Tensor,
    sols_pred: torch.Tensor,
    class_weights: Sequence,
    kld_weight: float,
):
    # the torch kl_divergence function actually calculates the NEGATIVE KLD
    kld = D.kl_divergence(posterior, prior).mean()

    sample_weights = torch.empty_like(sols)
    sample_weights[sols == 0] = class_weights[0]
    sample_weights[sols == 1] = class_weights[1]
    bce = F.binary_cross_entropy(sols_pred, sols, weight=sample_weights, reduction="mean")

    loss = kld_weight * kld + (1 - kld_weight) * bce

    metrics = {
        "kld": kld.detach(),
        "bce": bce.detach(),
        "loss": loss.detach(),
    }

    return loss, metrics


def train_step(model, solver, batch, device, train_set, kld_weight):
    batch_normed = train_set.norm(**batch._asdict())

    feats, costs, sols, _, _ = batch_normed
    feats = feats.to(device)
    costs = costs.to(device)
    sols = sols.to(device)

    prior, posterior = model(feats, sols)

    # when we are training, we are imitating the wait-and-see decision-making process, which is risk neutral
    # so we need to use the reparametrization trick
    y_pred = posterior.rsample()
    sols_pred = solver(y_pred)

    loss, train_metrics = get_train_metrics(prior, posterior, sols, sols_pred, train_set.class_weights, kld_weight)

    return loss, train_metrics


def eval_step(model, solver, batch, device, train_set, risk_level):
    feats, costs, sols, objs, cost_dist_params = batch
    feats = feats.to(device)
    costs = costs.to(device)
    sols = sols.to(device)
    objs = objs.to(device)
    cost_dist_mean, cost_dist_std = cost_dist_params.to(device).chunk(2, dim=-1)
    cost_dist = D.Normal(cost_dist_mean, cost_dist_std)

    feats_normed, _, _, _, _ = train_set.norm(feats=feats)

    prior = model.sample(feats_normed.to(device))

    if risk_level == 0.5:
        # we use the expectation for the risk neutral model
        y_pred = prior.loc
    else:
        # we use all parameters of the prior for the risk averse model
        y_pred = torch.cat([prior.loc, prior.scale], dim=-1)

    sols_pred = solver(y_pred)

    # in both cases, the objective is calculated with the expectation
    _, costs_pred, _, _, _ = train_set.unnorm(costs=prior.loc)
    objs_pred = torch.bmm(costs_pred.unsqueeze(1), sols_pred.unsqueeze(2)).squeeze(2)

    eval_metrics = get_val_metrics(
        costs,
        sols,
        objs,
        cost_dist,
        costs_pred,
        sols_pred,
        objs_pred,
        prior,
        data_model.modelSense,
        train_set.is_integer,
        train_set.class_weights,
    )

    return eval_metrics


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()

    args.use_wandb = args.wandb_project is not None
    if args.use_wandb:
        import wandb

        run_name = get_wandb_name(args, argparser)
        run = wandb.init(
            project=args.wandb_project,
            config=args,
            name=run_name,
            tags=["experiment"] + args.wandb_tags,
        )

    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"

    if args.dataset == "shortestpath":
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

    indices = torch.randperm(len(costs))
    train_indices, test_indices = train_test_split(indices, test_size=1000)
    train_set = PyEPODataset(data_model, feats[train_indices], costs[train_indices], cost_dist_params[train_indices])
    test_set = PyEPODataset(data_model, feats[test_indices], costs[test_indices], cost_dist_params[test_indices])

    train_bs, test_bs = min(args.batch_size, len(train_set)), min(args.batch_size, len(test_set))
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=args.workers, drop_last=True)

    feats, costs, sols, objs, cost_dist_params = train_set[0]
    model = SolverVAE(feats.size(-1), sols.size(-1), costs.size(-1), args.mlp_hidden_dim, args.mlp_hidden_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    imle = pyepo.func.implicitMLE(
        data_model, processes=args.workers, n_samples=16, solve_ratio=1.0, dataset=train_set, two_sides=True
    )
    if args.risk_level == 0.5:
        parallel_solver = ParallelSolver(args.workers, pyepo.model.grb.shortestPathModel, grid)
    else:
        parallel_solver = ParallelSolver(args.workers, CVaRShortestPath, grid, args.risk_level, tail="right")

    metrics = defaultdict(Average)
    metrics["val/obj_true"] = WandBHistogram()
    metrics["val/obj_pred"] = WandBHistogram()
    metrics["val/obj_expected"] = WandBHistogram()
    metrics["val/obj_realized"] = WandBHistogram()

    progress_bar = tqdm.trange(args.max_epochs)
    for epoch in progress_bar:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            loss, train_metrics = train_step(model, imle, batch, device, train_set, args.kld_weight)

            loss.backward()
            optimizer.step()

            for name, value in train_metrics.items():
                metrics["train/" + name].update(value)

        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eval_metrics = eval_step(model, parallel_solver, batch, device, train_set, args.risk_level)

                    for name, value in eval_metrics._asdict().items():
                        metrics["val/" + name].update(value)

        to_log = {}
        for name, avg in metrics.items():
            try:
                to_log[name] = avg.compute()
            except NotComputableError as e:
                pass

        if args.use_wandb:
            wandb.log(to_log, step=epoch)

            if epoch % args.save_every == 0:
                model_dir = os.environ.get("SLURM_TMPDIR", ".")
                model_path = pathlib.Path(model_dir) / "model.pt"
                name = run_name.replace(":", "_")
                alias = run_name.replace(":", "=") + f"_epoch={epoch}"
                torch.save(model.state_dict(), model_path)
                wandb.log_model(name=name, path=model_path, aliases=[alias])
        else:
            progress_bar.set_postfix(to_log)

        for avg in metrics.values():
            avg.reset()
