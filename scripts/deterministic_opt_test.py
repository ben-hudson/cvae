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

class ForwardOptimization(CMP):
    def __init__(self, dataset, device):
        super().__init__(is_constrained=True)
        self.device = device
        self.dataset = dataset

    def closure(self, model, batch):
        u_scaled, c_scaled, A_scaled, b_scaled, x_scaled, f_scaled = batch

        input = torch.cat([c_scaled, A_scaled.flatten(1), b_scaled], dim=1).to(self.device)
        x_pred_scaled = model(input).cpu()

        # lord, forgive me
        x_mean = torch.Tensor(self.dataset.x_scaler.mean_)
        x_scale = torch.Tensor(self.dataset.x_scaler.scale_)
        c_mean = torch.Tensor(self.dataset.c_scaler.mean_)
        c_scale = torch.Tensor(self.dataset.c_scaler.scale_)
        A_mean = torch.Tensor(self.dataset.A_scaler.mean_).reshape(A_scaled.size(1), A_scaled.size(2))
        A_scale = torch.Tensor(self.dataset.A_scaler.scale_).reshape(A_scaled.size(1), A_scaled.size(2))
        b_mean = torch.Tensor(self.dataset.b_scaler.mean_)
        b_scale = torch.Tensor(self.dataset.b_scaler.scale_)
        f_mean = torch.Tensor(self.dataset.f_scaler.mean_)
        f_scale = torch.Tensor(self.dataset.f_scaler.scale_)

        x_pred = x_pred_scaled*x_scale + x_mean
        x = x_scaled*x_scale + x_mean
        c = c_scaled*c_scale + c_mean
        A = A_scaled*A_scale + A_mean
        b = b_scaled*b_scale + b_mean
        f = f_scaled*f_scale + f_mean

        x_pred = x_pred.unsqueeze(2)
        x = x.unsqueeze(2)
        c = c.unsqueeze(1)
        b = b.unsqueeze(2)

        f_pred = torch.bmm(c, x_pred).squeeze(2)

        obj_loss = F.l1_loss(f, f_pred)
        constr_loss = torch.bmm(A, x_pred) - b

        return CMPState(loss=obj_loss, eq_defect=constr_loss)


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
    train_args.add_argument('--lr', type=float, default=1e-4, help='Optimizer learning rate')
    train_args.add_argument('--momentum', type=float, default=8e-2, help='Optimizer momentum')
    train_args.add_argument('--dual_restarts', action='store_true', help='Use dual restarts')
    train_args.add_argument('--extra_gradient', action='store_true', help='Use extra-gradient optimizers')
    train_args.add_argument('--max_epochs', type=int, default=500, help='Maximum number of training epochs')
    train_args.add_argument('--max_hours', type=int, default=3, help='Maximum hours to train')

    model_args = parser.add_argument_group('logging', description='Logging arguments')
    model_args.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    model_args.add_argument('--wandb_exp', type=str, default=None, help='WandB experiment name')
    model_args.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='WandB tags')

    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    args.use_wandb = args.wandb_project is not None
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
    model = MLP(in_dim, [128, 128, 64, 64, 32, 32, out_dim])
    model.to(device)

    # problem specific stuff happens in here
    cmp = ForwardOptimization(dataset, device)

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

            metrics = {
                "obj_loss": cmp.state.loss.item(),
                "constr_loss": cmp.state.eq_defect.sum().item()
            }
            progress_bar.set_postfix(metrics)
            if args.use_wandb:
                wandb.log(metrics)
