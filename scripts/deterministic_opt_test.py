import argparse
import cooper
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
import tqdm
import typing

from collections import namedtuple
from cooper import ConstrainedMinimizationProblem as CMP, CMPState
from itertools import pairwise
from torch.utils.data import DataLoader, Subset
from torchvision.ops import MLP

from synthetic_lp.dataset import SyntheticLPDataset

from ignite.metrics import RunningAverage

class ForwardOptimization(CMP):
    def __init__(self, dataset, device, target, nonneg_constr_method):
        super().__init__(is_constrained=True)
        self.device = device
        self.target = target
        self.nonneg_constr_method = nonneg_constr_method

        # lord, forgive me
        self.x_mean = torch.Tensor(dataset.x_scaler.mean_).to(self.device)
        self.x_scale = torch.Tensor(dataset.x_scaler.scale_).to(self.device)
        self.c_mean = torch.Tensor(dataset.c_scaler.mean_).to(self.device)
        self.c_scale = torch.Tensor(dataset.c_scaler.scale_).to(self.device)
        self.A_mean = torch.Tensor(dataset.A_scaler.mean_).to(self.device)
        self.A_scale = torch.Tensor(dataset.A_scaler.scale_).to(self.device)
        self.b_mean = torch.Tensor(dataset.b_scaler.mean_).to(self.device)
        self.b_scale = torch.Tensor(dataset.b_scaler.scale_).to(self.device)
        self.f_mean = torch.Tensor(dataset.f_scaler.mean_).to(self.device)
        self.f_scale = torch.Tensor(dataset.f_scaler.scale_).to(self.device)

    def closure(self, model, batch):
        u_scaled, c_scaled, A_scaled, b_scaled, x_scaled, f_scaled = batch
        u_scaled, c_scaled, A_scaled, b_scaled, x_scaled, f_scaled = u_scaled.to(device), c_scaled.to(device), A_scaled.to(device), b_scaled.to(device), x_scaled.to(device), f_scaled.to(device)

        input = torch.cat([c_scaled, A_scaled.flatten(1), b_scaled], dim=1)
        x_pred_scaled = model(input)

        x_pred = x_pred_scaled*self.x_scale + self.x_mean
        x = x_scaled*self.x_scale + self.x_mean
        c = c_scaled*self.c_scale + self.c_mean
        A = A_scaled*self.A_scale.reshape(A_scaled.size(1), A_scaled.size(2)) + self.A_mean.reshape(A_scaled.size(1), A_scaled.size(2))
        b = b_scaled*self.b_scale + self.b_mean
        f = f_scaled*self.f_scale + self.f_mean

        if self.nonneg_constr_method == 'softplus':
            x_pred = F.softplus(x_pred)
        elif self.nonneg_constr_method == 'relu':
            x_pred = F.relu(x_pred)

        x_pred = x_pred.unsqueeze(2)
        x = x.unsqueeze(2)
        c = c.unsqueeze(1)
        b = b.unsqueeze(2)

        f_pred = torch.bmm(c, x_pred).squeeze(2)

        aoe_loss = F.l1_loss(f, f_pred)
        obj_loss = f_pred.mean()

        # equality constraint Ax = b --> Ax - b = 0
        constr_loss = torch.bmm(A, x_pred) - b

        # non-negativity constraint x >= 0 --> -x <= 0
        if self.nonneg_constr_method == 'constr':
            nonneg_loss = -x_pred
        else:
            nonneg_loss = None

        if self.target == 'aoe':
            loss = aoe_loss
        elif self.target == 'f':
            loss = obj_loss
        else:
            raise Exception(f'unknown target {self.target}')

        metrics = {
            'aoe': aoe_loss.item(),
            'obj': obj_loss.item(),
            'constr_volation': constr_loss.mean().item(),
            'nonneg_loss': (-x_pred).mean().item()
        }

        return CMPState(loss=loss, ineq_defect=nonneg_loss, eq_defect=constr_loss, misc=metrics)


def get_datasets(args: namedtuple) -> typing.Tuple[Subset, Subset, Subset]:
    dataset = SyntheticLPDataset("datasets/2var8cons_InvPLP0.tar.gz", slack=True, norm=True)

    samples = len(dataset)
    indices = np.arange(samples).tolist() # dataloader shuffles, don't need to here
    split = np.array([0, args.train_frac, args.val_frac, args.test_frac])
    assert np.isclose(split.sum(), 1), f'train-val-test split does not sum to 1, it sums to {split.sum()}'

    split_indices = np.cumsum(split*samples).astype(int)
    subsets = tuple(Subset(dataset, indices[start:end]) for start, end in pairwise(split_indices))
    return subsets

def get_argparser():
    parser = argparse.ArgumentParser('Train an MLP to approximate an LP solver using constrained optimization')

    dataset_args = parser.add_argument_group('dataset', description='Dataset arguments')
    dataset_args.add_argument('dataset_path', type=pathlib.Path, help='Path to dataset')
    # dataset_args.add_argument('--train_frac', type=float, default=0.8, help='Fraction of dataset used for training')
    # dataset_args.add_argument('--val_frac', type=float, default=0.1, help='Fraction of dataset used for validation')
    # dataset_args.add_argument('--test_frac', type=float, default=0.1, help='Fraction of dataset used for testing')
    dataset_args.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    dataset_args.add_argument('--workers', type=int, default=2, help='Number of DataLoader workers')

    train_args = parser.add_argument_group('training', description='Training arguments')
    train_args.add_argument('--no_gpu', action='store_true', help='Do not use the GPU even if one is available')
    train_args.add_argument('--lr', type=float, default=1e-5, help='Optimizer learning rate')
    train_args.add_argument('--momentum', type=float, default=8e-2, help='Optimizer momentum')
    train_args.add_argument('--dual_restarts', action='store_true', help='Use dual restarts')
    train_args.add_argument('--extra_gradient', action='store_true', help='Use extra-gradient optimizers')
    train_args.add_argument('--max_epochs', type=int, default=500, help='Maximum number of training epochs')
    train_args.add_argument('--max_hours', type=int, default=3, help='Maximum hours to train')
    train_args.add_argument('--target', type=str, choices=['aoe', 'f'], default='aoe', help='Minimization target (aoe = l1 loss between predicted and true objective, f = true objective function)')
    train_args.add_argument('--nonneg_constr_method', type=str, choices=['none', 'softplus', 'relu', 'constr'], default='none', help='How to ensure predicted decisions satisfy non-negativity constraint')

    model_args = parser.add_argument_group('logging', description='Logging arguments')
    model_args.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    model_args.add_argument('--wandb_exp', type=str, default=None, help='WandB experiment name')
    model_args.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='WandB tags')

    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    args.use_wandb = args.wandb_project is not None
    assert not (args.target == 'f' and args.nonneg_constr_method == 'none'), 'You must specify a method to enforce the non-negativity constraint if the minimization target is f'
    # assert not (args.nonneg_constr_method != 'constr' and args.dual_restarts), 'Dual restarts will only have an effect if the non-negativity constraint is modelled as a constraint (our other constraints are equalities)'
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            config=args,
        )

    dataset = SyntheticLPDataset(args.dataset_path, slack=True, norm=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    device = 'cuda:0' if torch.cuda.is_available() and not args.no_gpu else 'cpu'

    u, c, A, b, x, f = dataset[0]
    in_dim = c.size(0) + A.size(0)*A.size(1) + b.size(0)
    out_dim = x.size(0)
    model = MLP(in_dim, [128, 128, 64, 64, 32, 32, out_dim], activation_layer=torch.nn.SiLU)
    model.to(device)

    # problem specific stuff happens in here
    cmp = ForwardOptimization(dataset, device, args.target, args.nonneg_constr_method)

    formulation = cooper.LagrangianFormulation(cmp)
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

    metrics = {
        'aoe': RunningAverage(output_transform=lambda x: x),
        'obj': RunningAverage(output_transform=lambda x: x),
        'constr_volation': RunningAverage(output_transform=lambda x: x),
        'nonneg_loss': RunningAverage(output_transform=lambda x: x)
    }

    progress_bar = tqdm.trange(args.max_epochs)
    for epoch in progress_bar:
        for batch in loader:
            model.train()
            optimizer.zero_grad()

            lagrangian = formulation.composite_objective(cmp.closure, model, batch)

            formulation.custom_backward(lagrangian)
            if args.extra_gradient:
                optimizer.step(cmp.closure, model, batch)
            else:
                optimizer.step()

            for name, avg in metrics.items():
                avg.update(cmp.state.misc[name])

        metrics_sample = {name: avg.compute() for name, avg in metrics.items()}
        progress_bar.set_postfix(metrics_sample)
        if args.use_wandb:
            wandb.log(metrics_sample)

        for avg in metrics.values():
            avg.reset()