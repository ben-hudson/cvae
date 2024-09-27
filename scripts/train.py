import argparse
import os
import pathlib
import pyepo
import pyepo.data
import pyepo.func
import pyepo.metric
import pyepo.model
import torch
import torch.distributions as D
import torch.nn.functional as F
import tqdm

from collections import defaultdict
from ignite.metrics import Average
from data.toy_example import gen_toy_data, get_toy_graph
from datasets.cso import CSLPDataset
from models.parallel_solver import ParallelSolver
from models.risk_averse import CVaRShortestPath
from models.shortestpath.risk_neutral import ILPShortestPath
from models.solver_vae import SolverVAE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Sequence
from utils.accumulator import Accumulator
from utils.eval import get_eval_metrics
from utils.wandb import get_friendly_name, record_metrics, save_metrics
from utils.utils import norm, norm_normal


def get_argparser():
    parser = argparse.ArgumentParser("Train an MLP to approximate an LP solver using constrained optimization")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads for multithreaded operations")

    dataset_args = parser.add_argument_group("dataset", description="Dataset arguments")
    dataset_args.add_argument("dataset", type=str, choices=["toy", "shortestpath"], help="Dataset to generate")
    dataset_args.add_argument("--n_samples", type=int, default=2000, help="Number of samples to generate")
    dataset_args.add_argument("--n_features", type=int, default=5, help="Number of features")
    dataset_args.add_argument("--degree", type=int, default=1, help="Polynomial degree for encoding function")
    dataset_args.add_argument("--noise_width", type=float, default=0.5, help="Half-width of latent uniform noise")

    model_args = parser.add_argument_group("model", description="Model arguments")
    model_args.add_argument("--mlp_hidden_dim", type=int, default=64, help="Dimension of hidden layers in MLPs")
    model_args.add_argument("--mlp_hidden_layers", type=int, default=2, help="Number of hidden layers in MLPs")
    model_args.add_argument("--latent_dist", type=str, default="normal", choices=["normal"], help="Latent distribution")
    model_args.add_argument("--latent_dim", type=int, default=10, help="Latent dimension")

    train_args = parser.add_argument_group("training", description="Training arguments")
    train_args.add_argument("--no_gpu", action="store_true", help="Do not use the GPU even if one is available")
    train_args.add_argument("--lr", type=float, default=1e-2, help="Optimizer learning rate")
    train_args.add_argument(
        "--kld_weight", type=float, default=0.99, help="Relative weighting of KLD and reconstruction loss"
    )
    train_args.add_argument(
        "--solver_perturb_temp", type=float, default=1.0, help="Temperature of solver input perturbation distribution"
    )
    train_args.add_argument(
        "--solver_perturb_samples", type=int, default=10, help="Samples to solve when computing solver gradient"
    )
    train_args.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")
    train_args.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    train_args.add_argument("--save_every", type=int, default=10, help="Save model weights every n epochs")

    eval_args = parser.add_argument_group("evaluation", description="Evaluation arguments")
    eval_args.add_argument("--eval_every", type=int, default=10, help="Evaluate every n epochs")
    eval_args.add_argument(
        "--risk_level", type=float, default=None, help="Risk level (probability) for risk-averse decision-making"
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
    batch_normed = train_set.norm(batch)

    feats, costs, sols, _, _ = batch_normed
    feats = feats.to(device)
    costs = costs.to(device)
    sols = sols.to(device)

    prior, posterior = model(feats, sols)

    # when we are training, we are imitating the wait-and-see decision-making process, which is risk agnostic
    # so we need to use the reparametrization trick
    y_sample = posterior.rsample()
    # norm the result to be consistent with other steps
    y_sample_normed = norm(y_sample)
    sols_pred = solver(y_sample_normed)

    loss, train_metrics = get_train_metrics(prior, posterior, sols, sols_pred, train_set.class_weights, kld_weight)

    return loss, train_metrics


def eval_step(model, data_model, solver, batch, device, train_set, risk_level):
    batch_normed = train_set.norm(batch)

    feats, costs, sols, objs, cost_dist = batch_normed
    feats = feats.to(device)
    costs = costs.to(device)
    sols = sols.to(device)
    objs = objs.to(device)
    cost_dist_mean, cost_dist_std = cost_dist_params.to(device).chunk(2, dim=-1)
    cost_dist = D.Normal(cost_dist_mean, cost_dist_std)

    feats_normed, _, _, _, _ = train_set.norm(feats=feats)

    cost_dist_pred = model.predict(feats_normed, point_prediction=False)
    costs_pred = cost_dist_pred.loc

    if risk_level is None:
        sols_pred = solver(cost_dist_pred.loc)
    else:
        cost_dist_pred_params = torch.cat([cost_dist_pred.loc, cost_dist_pred.scale], dim=-1)
        sols_pred = solver(cost_dist_pred_params)

    eval_metrics = get_eval_metrics(
        costs,
        sols,
        cost_dist,
        costs_pred,
        sols_pred,
        cost_dist_pred,
        data_model.modelSense,
        train_set.class_weights,
    )

    return eval_metrics


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()

    args.use_wandb = args.wandb_project is not None
    if args.use_wandb:
        import wandb

        run_name = get_friendly_name(args, argparser)
        run = wandb.init(
            project=args.wandb_project,
            config=args,
            name=run_name,
            tags=["experiment"] + args.wandb_tags,
        )

    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"

    if args.dataset == "toy":
        feats, costs, cost_dist, cost_dist_params = gen_toy_data(args.n_samples)
        graph = get_toy_graph()
        data_model = ILPShortestPath(graph, 0, 1)

    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    indices = torch.randperm(len(costs))
    train_ind, test_ind = train_test_split(indices, test_size=1000)
    train_set = CSLPDataset(data_model, feats[train_ind], costs[train_ind], cost_dist, cost_dist_params[train_ind])
    test_set = CSLPDataset(data_model, feats[test_ind], costs[test_ind], cost_dist, cost_dist_params[test_ind])

    train_bs, test_bs = min(args.batch_size, len(train_set)), min(args.batch_size, len(test_set))
    train_loader = DataLoader(
        train_set,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        collate_fn=train_set.collate_batch,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_bs,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
        collate_fn=test_set.collate_batch,
    )

    feats, costs, sols, objs, cost_dist_params = train_set[0]
    model = SolverVAE(feats.size(-1), sols.size(-1), costs.size(-1), args.mlp_hidden_dim, args.mlp_hidden_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    imle = pyepo.func.implicitMLE(
        data_model,
        processes=args.workers,
        n_samples=args.solver_perturb_samples,
        sigma=args.solver_perturb_temp,
        two_sides=True,
        solve_ratio=1.0,
        dataset=train_set,
    )
    if args.risk_level is None:
        parallel_solver = ParallelSolver(args.workers, ILPShortestPath, graph, 0, 1)
    else:
        parallel_solver = ParallelSolver(args.workers, CVaRShortestPath, graph, 0, 1, args.risk_level, tail="right")

    metrics = defaultdict(Average)
    metrics["val/obj_true"] = Accumulator()
    metrics["val/obj_pred"] = Accumulator()
    metrics["val/obj_expected"] = Accumulator()
    metrics["val/obj_realized"] = Accumulator()

    progress_bar = tqdm.trange(args.max_epochs)
    for epoch in progress_bar:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            loss, train_metrics = train_step(
                model, imle, batch, device, train_set, args.kld_weight, args.norm_latent_dists
            )

            loss.backward()
            optimizer.step()

            for name, value in train_metrics.items():
                metrics["train/" + name].update(value)

        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    eval_metrics = eval_step(
                        model,
                        data_model,
                        parallel_solver,
                        batch,
                        device,
                        train_set,
                        args.risk_level,
                        args.norm_latent_dists,
                    )

                    for name, value in eval_metrics._asdict().items():
                        metrics["val/" + name].update(value)

        if args.use_wandb:
            record_metrics(metrics, epoch)

            if epoch % args.save_every == 0:
                model_dir = os.environ.get("SLURM_TMPDIR", ".")
                model_path = pathlib.Path(model_dir) / "model.pt"
                name = run_name.replace(":", "_")
                alias = run_name.replace(":", "=") + f"_epoch={epoch}"
                torch.save(model.state_dict(), model_path)
                wandb.log_model(name=name, path=model_path, aliases=[alias])

        for avg in metrics.values():
            avg.reset()
