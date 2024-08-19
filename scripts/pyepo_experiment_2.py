import argparse
import pyepo
import pyepo.metric
import torch
import tqdm

from ignite.metrics import Average
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.ops import MLP

from data.pyepo import PyEPODataset, render_shortestpath
from metrics.util import get_val_metrics_sample
from models.solver_vae import SolverVAE


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

    model_args = parser.add_argument_group("model", description="Model arguments")
    model_args.add_argument("--model", type=str, choices=["nn", "cvae"], default="cvae", help="Model to predict costs")
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

    model_args = parser.add_argument_group("logging", description="Logging arguments")
    model_args.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    model_args.add_argument("--wandb_exp", type=str, default=None, help="WandB experiment name")
    model_args.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags")

    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    args.use_wandb = args.wandb_project is not None
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
    if args.model == "cvae":
        model = SolverVAE(feats.size(-1), sols.size(-1), costs.size(-1), args.mlp_hidden_dim, args.mlp_hidden_layers)
    elif args.model == "nn":
        hidden_layers = [args.mlp_hidden_dim] * args.mlp_hidden_layers
        model = MLP(feats.size(-1), hidden_layers + [costs.size(-1)], activation_layer=torch.nn.SiLU)
    else:
        raise ValueError()
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

            if args.model == "cvae":
                kld, costs_pred_normed = model(feats_normed, sols_normed)
            elif args.model == "nn":
                costs_pred_normed = model(feats_normed)
                kld = torch.tensor(0.0)
            else:
                raise ValueError()

            _, costs_pred, _, _ = train_set.unnorm(costs_normed=costs_pred_normed)

            spo = spo_loss(costs_pred, costs, sols, objs)
            loss = args.kld_weight * kld + (1 - args.kld_weight) * spo

            train_metrics = {
                "abs_obj": objs.detach().abs().mean(),
                "kld": kld.detach(),
                "loss": loss.detach(),
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

                if args.model == "cvae":
                    prior = model.sample(feats_normed.to(device))
                    costs_pred_normed = prior.loc
                elif args.model == "nn":
                    costs_pred_normed = model(feats_normed.to(device))
                else:
                    raise ValueError()
                _, costs_pred, _, _ = train_set.unnorm(costs_normed=costs_pred_normed)
                costs_pred = costs_pred.cpu()

                for i in range(len(costs)):
                    data_model.setObj(costs_pred[i])
                    sol_pred, obj_pred = data_model.solve()
                    sol_pred = torch.FloatTensor(sol_pred)

                    sample_metrics = get_val_metrics_sample(
                        data_model, costs[i], sols[i], objs[i], costs_pred[i], sol_pred, obj_pred, train_set.is_integer
                    )

                    for name, value in sample_metrics._asdict().items():
                        name = "val/" + name
                        if name not in metrics:
                            metrics[name] = Average()
                        metrics[name].update(value)

                img = torch.cat(
                    [render_shortestpath(data_model, sols[i]), render_shortestpath(data_model, sol_pred)], dim=0
                )
                imgs.append(img)

        if args.use_wandb:
            log = {name: avg.compute() for name, avg in metrics.items()}
            log["train/pyepo_regret_norm"] = log["train/spo_loss"] / (log["train/abs_obj"] + 1e-7)
            log["val/pyepo_regret_norm"] = log["val/spo_loss"] / (log["val/abs_obj"] + 1e-7)

            wandb.log(log, step=epoch)

            reconstruction = make_grid(torch.stack(imgs).permute(0, 3, 1, 2))
            wandb.log({"val/reconstruction": wandb.Image(reconstruction)})

        for avg in metrics.values():
            avg.reset()
