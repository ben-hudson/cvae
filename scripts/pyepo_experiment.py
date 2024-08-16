import argparse
import cooper
import pyepo
import pyepo.metric
import torch
import tqdm

from ignite.metrics import RunningAverage
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.pyepo import PyEPODataset
from trainers.portfolio import PortfolioTrainer
from models.cvae import CVAE
from trainers.lp import LPTrainer


def get_argparser():
    parser = argparse.ArgumentParser("Train an MLP to approximate an LP solver using constrained optimization")

    dataset_args = parser.add_argument_group("dataset", description="Dataset arguments")
    dataset_args.add_argument("dataset", type=str, choices=["shortestpath", "portfolio"], help="Dataset to generate")
    dataset_args.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate")
    dataset_args.add_argument("--n_features", type=int, default=5, help="Number of features")
    dataset_args.add_argument("--degree", type=int, default=1, help="Polynomial degree for encoding function")
    dataset_args.add_argument("--noise_width", type=float, default=0.5, help="Half-width of latent uniform noise")
    dataset_args.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    dataset_args.add_argument("--workers", type=int, default=2, help="Number of DataLoader workers")

    model_args = parser.add_argument_group("model", description="Model arguments")
    model_args.add_argument("--mlp_hidden_dim", type=int, default=64, help="Dimension of hidden layers in MLPs")
    model_args.add_argument("--mlp_hidden_layers", type=int, default=2, help="Number of hidden layers in MLPs")
    model_args.add_argument(
        "--latent_dist", type=str, default="normal", choices=["normal", "uniform"], help="Latent distribution"
    )
    model_args.add_argument("--latent_dim", type=int, default=10, help="Latent dimension")
    model_args.add_argument(
        "--costs_are_latents", action="store_true", help="Costs are the latents (overrides latent_dim)"
    )

    train_args = parser.add_argument_group("training", description="Training arguments")
    train_args.add_argument("--no_gpu", action="store_true", help="Do not use the GPU even if one is available")
    train_args.add_argument("--lr", type=float, default=1e-5, help="Optimizer learning rate")
    train_args.add_argument(
        "--kld_weight", type=float, default=0.01, help="Relative weighting of KLD and reconstruction loss"
    )
    train_args.add_argument("--momentum", type=float, default=8e-2, help="Optimizer momentum")
    train_args.add_argument("--max_epochs", type=int, default=500, help="Maximum number of training epochs")
    train_args.add_argument("--max_hours", type=int, default=3, help="Maximum hours to train")
    train_args.add_argument("--dual_restarts", action="store_true", help="Use dual restarts")
    train_args.add_argument("--no_extra_gradient", action="store_true", help="Use extra-gradient optimizers")

    model_args = parser.add_argument_group("logging", description="Logging arguments")
    model_args.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    model_args.add_argument("--wandb_exp", type=str, default=None, help="WandB experiment name")
    model_args.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags")

    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    args.use_wandb = args.wandb_project is not None
    args.extra_gradient = not args.no_extra_gradient

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            config=args,
        )

    device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"

    if args.dataset == "portfolio":
        n_assets = 50
        gamma = 2.25
        cov, feats, costs = pyepo.data.portfolio.genData(
            args.n_samples, args.n_features, n_assets, deg=args.degree, noise_level=args.noise_width, seed=135
        )
        data_model = pyepo.model.grb.portfolioModel(n_assets, cov, gamma)

    elif args.dataset == "shortestpath":
        grid = (5, 5)
        feats, costs = pyepo.data.shortestpath.genData(
            args.n_samples, args.n_features, grid, deg=args.degree, noise_width=args.noise_width, seed=135
        )
        data_model = pyepo.model.grb.shortestPathModel(grid)

    else:
        raise ValueError("NYI")

    indices = torch.randperm(len(costs))
    train_indices, test_indices = train_test_split(indices, test_size=1000)
    train_set = PyEPODataset(data_model, feats[train_indices], costs[train_indices], norm=True)
    test_set = PyEPODataset(data_model, feats[test_indices], costs[test_indices], norm=False)

    train_bs, test_bs = min(args.batch_size, len(train_set)), min(args.batch_size, len(test_set))
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=args.workers, drop_last=True)

    feats, costs, sols, objs = train_set[0]
    if args.costs_are_latents:
        latent_dim = costs.size(-1)
        obs_dim = sols.size(-1)
    else:
        latent_dim = args.latent_dim
        obs_dim = costs.size(-1) + sols.size(-1)
    model = CVAE(feats.size(-1), obs_dim, args.mlp_hidden_dim, args.mlp_hidden_layers, latent_dim, args.latent_dist)
    model.to(device)

    if args.dataset == "portfolio":
        trainer = PortfolioTrainer(
            train_set.unnorm,
            device,
            data_model,
            costs_are_latents=args.costs_are_latents,
            kld_weight=args.kld_weight,
        )

    elif args.dataset == "shortestpath":
        trainer = LPTrainer(
            train_set.unnorm,
            device,
            data_model,
            is_integer=train_set.is_integer,
            costs_are_latents=args.costs_are_latents,
            kld_weight=args.kld_weight,
        )

    else:
        raise ValueError("NYI")

    formulation = cooper.LagrangianFormulation(trainer)
    if args.extra_gradient:
        primal_optimizer = cooper.optim.ExtraAdam(model.parameters(), lr=args.lr)
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam, lr=args.lr)
    else:
        primal_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        dual_optimizer = cooper.optim.partial_optimizer(torch.optim.Adam, lr=args.lr)

    optimizer = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=args.dual_restarts,
    )

    # these get populated automatically
    metrics = {}

    progress_bar = tqdm.trange(args.max_epochs)
    for epoch in progress_bar:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            lagrangian = formulation.composite_objective(trainer.closure, model, batch)

            formulation.custom_backward(lagrangian)
            if args.extra_gradient:
                optimizer.step(trainer.closure, model, batch)
            else:
                optimizer.step()

            for name, value in trainer.state.misc.items():
                name = "train/" + name
                if name not in metrics:
                    metrics[name] = RunningAverage(output_transform=lambda x: x)
                metrics[name].update(value)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                val_metrics = trainer.val(model, batch)

                for name, value in val_metrics.items():
                    name = "val/" + name
                    if name not in metrics:
                        metrics[name] = RunningAverage(output_transform=lambda x: x)
                    metrics[name].update(value)

        if args.use_wandb:
            log = {name: avg.compute() for name, avg in metrics.items()}
            log["train/pyepo_regret_norm"] = log["train/total_pyepo_regret"] / (log["train/total_obj"] + 1e-7)
            log["val/pyepo_regret_norm"] = log["val/total_pyepo_regret"] / (log["val/total_obj"] + 1e-7)

            wandb.log(log, step=epoch)

        for avg in metrics.values():
            avg.reset()
