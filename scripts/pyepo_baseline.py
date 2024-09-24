import argparse
import json
import pathlib
import pyepo
import pyepo.metric
import torch
import torch.utils
import tqdm

from collections import defaultdict
from data.pyepo import PyEPODataset, gen_shortestpath_data
from ignite.metrics import Average
from models.parallel_solver import ParallelSolver
from models.risk_averse import CVaRShortestPath
from models.solver_vae import SolverVAE
from utils.accumulator import Accumulator
from utils.eval import get_eval_metrics
from utils.wandb import get_friendly_name, record_metrics, save_metrics


def get_argparser():
    parser = argparse.ArgumentParser("Run a baseline on a dataset")

    # we need to keep default=None for all arguments so we can tell if we should override them from the config values
    # effectively, the config values become defaults
    parser.add_argument(
        "--config", type=pathlib.Path, help="WandB experiment config to reproduce (additional arguments override)"
    )

    dataset_args = parser.add_argument_group("dataset", description="Dataset arguments")
    dataset_args.add_argument("--dataset", type=str, choices=["shortestpath"], help="Dataset to generate")
    dataset_args.add_argument("--n_samples", type=int, help="Number of samples to generate")
    dataset_args.add_argument("--n_features", type=int, help="Number of features")
    dataset_args.add_argument("--degree", type=int, help="Polynomial degree for encoding function")
    dataset_args.add_argument("--noise_width", type=float, help="Half-width of latent uniform noise")
    dataset_args.add_argument("--workers", type=int, help="Number of DataLoader workers")
    dataset_args.add_argument("--seed", type=int, help="RNG seed")

    model_args = parser.add_argument_group("model", description="Model arguments")
    model_args.add_argument(
        "--model", type=str, choices=["oracle", "random", "mean", "pretrained"], help="Prediction model"
    )
    model_args.add_argument("--weights", type=pathlib.Path, help="Weights for pretrained prediction model")

    eval_args = parser.add_argument_group("evaluation", description="Evaluation arguments")
    eval_args.add_argument("--risk_level", type=float, help="Risk level (probability) for risk-averse decision-making")

    train_args = parser.add_argument_group("training", description="Training arguments")
    train_args.add_argument("--max_epochs", type=int, help="Maximum number of training epochs")

    model_args = parser.add_argument_group("logging", description="Logging arguments")
    model_args.add_argument("--wandb_project", type=str, help="WandB project name")
    model_args.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags")

    return parser


def set_args_from_config(args, config):
    args_dict = vars(args)
    for arg, value in args_dict.items():
        if arg in config and argparser.get_default(arg) == value:
            new_value = config[arg]["value"]
            print(f"setting {arg}={new_value} from config")
            args_dict[arg] = new_value
    return args


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    if args.config is not None:
        config = json.load(open(args.config))
        args = set_args_from_config(args, config)

    args.use_wandb = args.wandb_project is not None
    if args.use_wandb:
        import wandb

        run_name = get_friendly_name(args, argparser)

        if args.model is not None:
            tags = ["experiment"]
        elif args.baseline is not None:
            tags = ["baseline"]

        run = wandb.init(
            project=args.wandb_project,
            config=args,
            name=run_name,
            tags=tags + args.wandb_tags,
        )

    torch.manual_seed(args.seed)

    # to generate the "oracle" predictions we generate the data with no noise, and then add noise back in to get the observations
    assert args.noise_width <= 1, "noise width must be at most 1"
    if args.dataset == "shortestpath":
        is_integer = True
        grid = (5, 5)
        feats, costs, cost_dist_params = gen_shortestpath_data(
            args.n_samples, args.n_features, grid, args.degree, args.noise_width, args.seed
        )
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

    if args.model == "pretrained":
        assert args.config is not None, "--config must be specified when model=pretrained"
        assert args.weights is not None, "--weights must be specified when model=pretrained"

        feats, costs, sols, objs, cost_dist_params = dataset[0]
        model = SolverVAE(
            feats.size(-1),
            sols.size(-1),
            costs.size(-1),
            config["mlp_hidden_dim"]["value"],
            config["mlp_hidden_layers"]["value"],
        )
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
        model.eval()

    if args.risk_level == 0.5:
        solver = ParallelSolver(args.workers, pyepo.model.grb.shortestPathModel, grid)
    else:
        solver = ParallelSolver(args.workers, CVaRShortestPath, grid, args.risk_level, tail="right")

    metrics = defaultdict(Average)
    # override these
    metrics["val/obj_true"] = Accumulator()
    metrics["val/obj_pred"] = Accumulator()
    metrics["val/obj_expected"] = Accumulator()
    metrics["val/obj_realized"] = Accumulator()

    for epoch, batch in tqdm.tqdm(enumerate(dataloader), total=args.max_epochs):
        feats, costs, sols, objs, cost_dist_params = batch
        cost_dist_mean, cost_dist_std = cost_dist_params.chunk(2, dim=-1)
        cost_dist = torch.distributions.Normal(cost_dist_mean, cost_dist_std)

        if args.model == "pretrained":
            with torch.no_grad():
                feats_normed, _, _, _, _ = dataset.norm(feats=feats)
                prior_normed = model.predict(feats_normed, point_prediction=False)
                if args.risk_level == 0.5:
                    y_pred = prior_normed.loc
                else:
                    y_pred = torch.cat([prior_normed.loc, prior_normed.scale], dim=-1)
                costs_pred = prior_normed.loc

        elif args.model == "oracle":
            if args.risk_level == 0.5:
                y_pred = cost_dist_mean
            else:
                y_pred = cost_dist_params
            costs_pred = cost_dist_mean

        elif args.model == "mean":
            costs_pred_unnorm = dataset.means.costs.expand_as(costs)
            costs_pred_norm = costs_pred_unnorm.norm(dim=-1).unsqueeze(-1)
            costs_pred = costs_pred_unnorm / costs_pred_norm
            if args.risk_level == 0.5:
                y_pred = costs_pred
            else:
                y_pred = torch.cat([costs_pred, dataset.scales.costs.expand_as(costs) / costs_pred_norm], dim=-1)

        elif args.model == "random":
            costs_pred_unnorm = dataset.means.costs + torch.rand_like(costs) * dataset.scales.costs
            costs_pred = costs_pred_unnorm / costs_pred_unnorm.norm(dim=-1).unsqueeze(-1)
            if args.risk_level == 0.5:
                y_pred = costs_pred
            else:
                raise ValueError("no risk averse random baseline")

        else:
            raise ValueError(f"unknown baseline {args.model}")

        sols_pred = solver(y_pred)

        objs_pred = torch.bmm(costs_pred.unsqueeze(1), sols_pred.unsqueeze(2)).squeeze(2)

        eval_metrics = get_eval_metrics(
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

        if args.use_wandb:
            record_metrics(metrics, epoch)

    if args.use_wandb:
        save_metrics(metrics, args.max_epochs)
