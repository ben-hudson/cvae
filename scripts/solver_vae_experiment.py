import argparse
import pyepo
import pyepo.metric
import torch
import tqdm

from ignite.metrics import Average
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.pyepo import PyEPODataset
from models.ccsp import ChanceConstrainedShortestPath
from models.solver_vae import SolverVAE
from utils import get_sample_val_metrics, get_wandb_name


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

    eval_args = parser.add_argument_group("evaluation", description="Evaluation arguments")
    eval_args.add_argument(
        "--chance_constraint_budget", type=float, default=None, help="Chance constraint cost threshold"
    )
    eval_args.add_argument(
        "--chance_constraint_prob", type=float, default=0.5, help="Chance constraint probability threshold"
    )

    model_args = parser.add_argument_group("logging", description="Logging arguments")
    model_args.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    model_args.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags")

    return parser


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
            tags=["experiment"] + args.wandb_tags,
        )

    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu"

    if args.dataset == "shortestpath":
        grid = (5, 5)
        feats, costs = pyepo.data.shortestpath.genData(
            args.n_samples, args.n_features, grid, deg=args.degree, noise_width=args.noise_width, seed=args.seed
        )
        data_model = pyepo.model.grb.shortestPathModel(grid)

    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    indices = torch.randperm(len(costs))
    train_indices, test_indices = train_test_split(indices, test_size=1000)
    train_set = PyEPODataset(data_model, feats[train_indices], costs[train_indices], norm=True)
    test_set = PyEPODataset(data_model, feats[test_indices], costs[test_indices], norm=False)

    train_bs, test_bs = min(args.batch_size, len(train_set)), min(args.batch_size, len(test_set))
    train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=args.workers, drop_last=True)

    feats, costs, sols, objs = train_set[0]
    model = SolverVAE(feats.size(-1), sols.size(-1), costs.size(-1), args.mlp_hidden_dim, args.mlp_hidden_layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    spo_loss = pyepo.func.SPOPlus(data_model, processes=args.workers)

    # these get populated automatically
    metrics = {}

    progress_bar = tqdm.trange(args.max_epochs)
    for epoch in progress_bar:
        for batch_normed in train_loader:
            model.train()
            optimizer.zero_grad()

            feats_normed, costs_normed, sols_normed, objs_normed = batch_normed
            feats_normed = feats_normed.to(device)
            costs_normed = costs_normed.to(device)
            sols_normed = sols_normed.to(device)
            objs_normed = objs_normed.to(device)

            feats, costs, sols, objs = train_set.unnorm(feats_normed, costs_normed, sols_normed, objs_normed)

            kld, costs_pred_normed = model(feats_normed, sols_normed)

            _, costs_pred, _, _ = train_set.unnorm(costs_normed=costs_pred_normed)

            spo = spo_loss(costs_pred, costs, sols, objs)
            loss = args.kld_weight * kld + (1 - args.kld_weight) * spo

            train_metrics = {
                "kld": kld.detach(),
                "loss": loss.detach(),
                "obj_true": objs.detach().abs().mean(),
                "spo_loss": spo.detach().mean(),
            }

            loss.backward()
            optimizer.step()

            for name, value in train_metrics.items():
                name = "train/" + name
                if name not in metrics:
                    metrics[name] = Average()
                metrics[name].update(value)

        model.eval()
        imgs = []
        with torch.no_grad():
            for batch in test_loader:
                feats, costs, sols, objs = batch
                feats_normed, costs_normed, sols_normed, objs_normed = train_set.norm(*batch)

                prior_normed = model.sample(feats_normed.to(device))
                _, costs_pred, _, _ = train_set.unnorm(costs_normed=prior_normed.loc.cpu())
                # we have to unnorm the std as well, which is just scale*std
                # from https://en.wikipedia.org/wiki/Normal_distribution#Operations_and_functions_of_normal_variables
                costs_pred_std = prior_normed.scale.cpu() * train_set.scales.costs
                prior = torch.distributions.Normal(costs_pred, costs_pred_std)

                # solve the problems explicitly
                for i in range(len(costs)):
                    try:
                        if args.chance_constraint_prob == 0.5:
                            # if the decision-making is risk-neutral just use the normal model
                            data_model.setObj(prior.loc[i])
                            sol_pred, obj_pred = data_model.solve()

                        elif args.chance_constraint_prob > 0.5:
                            if args.dataset == "shortestpath":
                                cc_data_model = ChanceConstrainedShortestPath(
                                    grid,
                                    prior.loc[i],
                                    prior.scale[i],
                                    args.chance_constraint_budget,
                                    args.chance_constraint_prob,
                                )
                                sol_pred, obj_pred = cc_data_model.solve()
                            else:
                                raise ValueError(f"unknown dataset {args.dataset}")

                        else:
                            raise ValueError("alpha must be at least 0.5 (0.5 corresponds to a risk-neutral decision)")

                        sol_pred = torch.FloatTensor(sol_pred)

                        sample_metrics = get_sample_val_metrics(
                            data_model,
                            costs[i],
                            sols[i],
                            objs[i],
                            costs_pred[i],
                            sol_pred,
                            obj_pred,
                            train_set.is_integer,
                        )

                        for name, value in sample_metrics._asdict().items():
                            name = "val/" + name
                            if name not in metrics:
                                metrics[name] = Average()
                            metrics[name].update(value)

                    except:
                        if "val/success" not in metrics:
                            metrics["val/success"] = Average()
                        metrics["val/success"].update(0.0)

        if args.use_wandb:
            log = {name: avg.compute() for name, avg in metrics.items()}
            if "train/spo_loss" in log and "train/obj_true" in log:
                log["train/pyepo_regret_norm"] = log["train/spo_loss"] / (log["train/obj_true"] + 1e-7)
            if "val/spo_loss" in log and "val/obj_true" in log:
                log["val/pyepo_regret_norm"] = log["val/spo_loss"] / (log["val/obj_true"] + 1e-7)

            wandb.log(log, step=epoch)

        for avg in metrics.values():
            avg.reset()
