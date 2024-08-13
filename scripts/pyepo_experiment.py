import argparse
import cooper
import pyepo
import pyepo.metric
import torch
import tqdm

from cooper import ConstrainedMinimizationProblem as CMP, CMPState
from ignite.metrics import RunningAverage
from sklearn.model_selection import train_test_split
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from data.portfolio import PortfolioDataset
from models.cvae import CVAE
from metrics import mcc, r2

class PortfolioTrainer(CMP):
    def __init__(self, unnorm_func, device, portfolio_model, costs_are_latents=False):
        super().__init__(is_constrained=True)

        self.unnorm = unnorm_func
        self.device = device
        self.portfolio_model = portfolio_model
        self.costs_are_latents = costs_are_latents

    def closure(self, model, batch):
        feats_normed, costs_normed, sols_normed, objs_normed = batch
        feats_normed = feats_normed.to(self.device)
        costs_normed = costs_normed.to(self.device)
        sols_normed = sols_normed.to(self.device)
        objs_normed = objs_normed.to(self.device)

        if self.costs_are_latents:
            prior, posterior, costs_hat_normed, sols_hat_normed = model(feats_normed, sols_normed)
        else:
            obs_normed = torch.cat([costs_normed, sols_normed], dim=-1)
            prior, posterior, _, obs_hat = model(feats_normed, obs_normed)
            costs_hat_normed, sols_hat_normed = obs_hat.split([costs_normed.size(-1), sols_normed.size(-1)], dim=-1)

        _, costs_hat, sols_hat, _ = self.unnorm(None, costs_hat_normed, sols_hat_normed, None)
        _, costs, sols, objs = self.unnorm(feats_normed, costs_normed, sols_normed, objs_normed)

        kld = kl_divergence(prior, posterior).mean()

        sols = sols.unsqueeze(2)
        costs = costs.unsqueeze(1)
        sols_hat = sols_hat.unsqueeze(2)
        costs_hat = costs_hat.unsqueeze(1)
        objs_hat = torch.bmm(costs_hat, sols_hat).squeeze(2)

        regret = torch.bmm(costs, sols_hat).squeeze(2) - objs
        satisfaction = torch.bmm(costs_hat, sols).squeeze(2) - objs # the opposite of regret, haha

        eq_constrs = torch.stack([
            objs_hat - objs,
            regret,
            satisfaction,
        ], dim=1)

        # regret is the other way around when maximizing
        if self.portfolio_model.modelSense == pyepo.EPO.MAXIMIZE:
            eq_constrs *= -1

        budget = 1 # budget is hardcoded to 1
        budget_constr = sols_hat.sum(axis=1) - budget

        cov = torch.FloatTensor(self.portfolio_model.covariance).to(self.device).expand(sols_hat.size(0), -1, -1) # create batch of covariance matrices
        risk_constr = torch.bmm(sols_hat.permute(0, 2, 1), torch.bmm(cov, sols_hat)).squeeze(2) - self.portfolio_model.risk_level

        ineq_constrs = torch.stack([
            budget_constr,
            risk_constr,
        ], dim=1)

        metrics = {
            "kld": kld.detach(),
            "regret": regret.detach().mean(),
            "budget_viol": budget_constr.detach().clamp(min=0).mean(),
            "risk_viol": risk_constr.detach().clamp(min=0).mean(),
        }

        return CMPState(loss=kld, ineq_defect=ineq_constrs, eq_defect=eq_constrs, misc=metrics)

    def val(self, model, batch):
        feats_normed, costs_normed, sols_normed, objs_normed = batch
        prior, latents_hat, obs_hat = model.sample(feats_normed.to(self.device), mean=True)

        if self.costs_are_latents:
            costs_hat_normed = latents_hat
        else:
            costs_hat_normed, _ = obs_hat.split([costs_normed.size(-1), sols_normed.size(-1)], dim=-1)
        costs_hat_normed = costs_hat_normed.cpu()

        _, costs_hat, _, _ = self.unnorm(costs_normed=costs_hat_normed)
        _, costs, _, objs = self.unnorm(costs_normed=costs_normed, objs_normed=objs_normed)

        total_regret = 0
        total_obj = 0
        for i in range(costs.size(-1)):
            obj = objs[i].item()
            total_regret += pyepo.metric.calRegret(self.portfolio_model, costs_hat[i], costs[i], obj)
            total_obj += abs(obj)

        metrics = {
            "mcc": mcc(costs_hat_normed, costs_normed),
            "r2": r2(costs_hat_normed, costs_normed),
            "total_regret": total_regret,
            "total_obj": total_obj,
        }

        return metrics

def render_shortestpath(obs, grid_size):
    grid = torch.zeros(grid_size, dtype=torch.float32)

    index = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if i != j:
                grid[i, j] = obs[index]
                index += 1

    return grid

def get_argparser():
    parser = argparse.ArgumentParser('Train an MLP to approximate an LP solver using constrained optimization')

    dataset_args = parser.add_argument_group('dataset', description='Dataset arguments')
    dataset_args.add_argument('dataset', type=str, choices=['portfolio'], help='Dataset to generate')
    dataset_args.add_argument('--n_samples', type=int, default=100000, help='Number of samples to generate')
    dataset_args.add_argument('--n_features', type=int, default=5, help='Number of features')
    dataset_args.add_argument('--degree', type=int, default=1, help='Polynomial degree for encoding function')
    dataset_args.add_argument('--noise_width', type=float, default=0.5, help='Half-width of latent uniform noise')
    dataset_args.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    dataset_args.add_argument('--workers', type=int, default=2, help='Number of DataLoader workers')

    model_args = parser.add_argument_group('model', description='Model arguments')
    model_args.add_argument('--latent_dist', type=str, default='normal', choices=['normal', 'uniform'], help='Latent distribution')
    model_args.add_argument('--latent_dim', type=int, default=10, help='Latent dimension')
    model_args.add_argument('--costs_are_latents', action='store_true', help='Costs are the latents (overrides latent_dim)')

    train_args = parser.add_argument_group('training', description='Training arguments')
    train_args.add_argument('--no_gpu', action='store_true', help='Do not use the GPU even if one is available')
    train_args.add_argument('--lr', type=float, default=1e-5, help='Optimizer learning rate')
    train_args.add_argument('--kld_weight', type=float, default=0.01, help='Relative weighting of KLD and reconstruction loss')
    train_args.add_argument('--momentum', type=float, default=8e-2, help='Optimizer momentum')
    train_args.add_argument('--max_epochs', type=int, default=500, help='Maximum number of training epochs')
    train_args.add_argument('--max_hours', type=int, default=3, help='Maximum hours to train')
    # cooper related args
    train_args.add_argument('--dual_restarts', action='store_true', help='Use dual restarts')
    train_args.add_argument('--no_extra_gradient', action='store_true', help='Use extra-gradient optimizers')


    model_args = parser.add_argument_group('logging', description='Logging arguments')
    model_args.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    model_args.add_argument('--wandb_exp', type=str, default=None, help='WandB experiment name')
    model_args.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='WandB tags')

    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    args.use_wandb = args.wandb_project is not None
    args.extra_gradient = not args.no_extra_gradient

    if args.use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            config=args,
        )

    device = 'cuda:0' if torch.cuda.is_available() and not args.no_gpu else 'cpu'

    if args.dataset == 'portfolio':
        n_assets = 50
        gamma = 2.25
        cov, feats, costs = pyepo.data.portfolio.genData(args.n_samples, args.n_features, n_assets, deg=args.degree, noise_level=args.noise_width, seed=135)
        portfolio_model = pyepo.model.grb.portfolioModel(n_assets, cov, gamma)

        indices = torch.randperm(len(costs))
        train_indices, test_indices = train_test_split(indices, test_size=1000)
        train_set = PortfolioDataset(portfolio_model, feats[train_indices], costs[train_indices], norm=True)
        test_set = PortfolioDataset(portfolio_model, feats[test_indices], costs[test_indices], norm=False)
        train_loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True, num_workers=args.workers, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=min(args.batch_size, len(test_set)), shuffle=False, num_workers=args.workers, drop_last=True)

        trainer = PortfolioTrainer(train_set.unnorm, device, portfolio_model, costs_are_latents=args.costs_are_latents)

    else:
        raise ValueError('NYI')

    feats, costs, sols, objs = train_set[0]
    if args.costs_are_latents:
        latent_dim = costs.size(-1)
        obs_dim = sols.size(-1)
    else:
        latent_dim = args.latent_dim
        obs_dim = costs.size(-1) + sols.size(-1)
    model = CVAE(feats.size(-1), obs_dim, latent_dim, args.latent_dist)
    model.to(device)

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
        dual_restarts=args.dual_restarts
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
                name = 'train/' + name
                if name not in metrics:
                    metrics[name] = RunningAverage(output_transform=lambda x: x)
                metrics[name].update(value)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                val_metrics = trainer.val(model, batch)

                for name, value in val_metrics.items():
                    name = 'val/' + name
                    if name not in metrics:
                        metrics[name] = RunningAverage(output_transform=lambda x: x)
                    metrics[name].update(value)

        if args.use_wandb:
            log = {name: avg.compute() for name, avg in metrics.items()}
            log['val/regret_norm'] = log['val/total_regret'] / (log['val/total_obj'] + 1e-7)

            wandb.log(log, step=epoch)

        for avg in metrics.values():
            avg.reset()
