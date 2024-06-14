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
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA

from synthetic_lp.dataset import SyntheticLPDataset

from ignite.metrics import RunningAverage

from models.deterministic_lp import DeterministicLP

class ForwardOptimization(CMP):
    def __init__(self, dataset, device, target, nonneg_constr_method, constr_type):
        super().__init__(is_constrained=True)
        self.device = device
        self.target = target
        self.nonneg_constr_method = nonneg_constr_method
        self.constr_type = constr_type

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

    def sample(self, model, batch):
        u_scaled, c_scaled, A_scaled, b_scaled, x_scaled, f_scaled = batch
        u_scaled, c_scaled, A_scaled, b_scaled, x_scaled, f_scaled = u_scaled.to(device), c_scaled.to(device), A_scaled.to(device), b_scaled.to(device), x_scaled.to(device), f_scaled.to(device)

        c_pred_scaled, A_pred_scaled, b_pred_scaled, x_pred_scaled = model(u_scaled)

        x = x_scaled*self.x_scale + self.x_mean
        c = c_scaled*self.c_scale + self.c_mean
        A = A_scaled*self.A_scale.reshape(A_scaled.size(1), A_scaled.size(2)) + self.A_mean.reshape(A_scaled.size(1), A_scaled.size(2))
        b = b_scaled*self.b_scale + self.b_mean
        f = f_scaled*self.f_scale + self.f_mean

        x = x.unsqueeze(2)
        c = c.unsqueeze(1)
        b = b.unsqueeze(2)

        x_pred = x_pred_scaled*self.x_scale + self.x_mean
        c_pred = c_pred_scaled*self.c_scale + self.c_mean
        A_pred = A_pred_scaled*self.A_scale.reshape(A_pred_scaled.size(1), A_pred_scaled.size(2)) + self.A_mean.reshape(A_pred_scaled.size(1), A_pred_scaled.size(2))
        b_pred = b_pred_scaled*self.b_scale + self.b_mean

        x_pred = x_pred.unsqueeze(2)
        c_pred = c_pred.unsqueeze(1)
        b_pred = b_pred.unsqueeze(2)
        f_pred = torch.bmm(c_pred, x_pred).squeeze(2)

        return (c, A, b, x, f), (c_pred, A_pred, b_pred, x_pred, f_pred)

    def closure(self, model, batch):
        (c, A, b, x, f), (c_pred, A_pred, b_pred, x_pred, f_pred) = self.sample(model, batch)

        f_c_pred_x_gt = torch.bmm(c_pred, x).squeeze(2)
        f_c_gt_x_pred = torch.bmm(c, x_pred).squeeze(2)

        # so we observe x and f, and we want to thse to get losses on all of the constraints
        # A_pred*x_pred = b_pred
        # A_pred*x = b_pred
        # c_pred*x = c_pred*x_pred = f
        constr_loss = torch.bmm(A_pred, x_pred) - b_pred
        gt_constr_loss = torch.bmm(A_pred, x) - b_pred
        if self.constr_type == 'ineq':
            ineq_constr_list = [constr_loss.squeeze(2), gt_constr_loss.squeeze(2)]
            eq_constr_list = []
        elif self.constr_type == 'eq':
            eq_constr_list = [constr_loss.squeeze(2), gt_constr_loss.squeeze(2)]
            ineq_constr_list = []

        # if self.target != 'aoe': # if we are minimizing aoe we don't need to add the constraint too
        #     eq_constr_list.append(f_pred - f)
        # if self.target != 'aoe_c_only': # likewise for the cost-only aoe
        #     eq_constr_list.append(f_c_pred_x_gt - f)
        # if self.target != 'aoe_x_only': # likewise for the cost-only aoe
        #     eq_constr_list.append(f_c_gt_x_pred - f)

        ineq_constrs = torch.zeros(A.size(0), 1) if len(ineq_constr_list) == 0 else torch.cat(ineq_constr_list, dim=1)
        eq_constrs = torch.zeros(A.size(0), 1) if len(eq_constr_list) == 0 else torch.cat(eq_constr_list, dim=1)


        # but which minimization objective do we choose?
        # min c_pred*x_pred
        # min c_pred*x
        # min l1(f, c_pred*x)
        # min l1(f, c_pred*x_pred)
        if self.target == 'aoe':
            loss = F.l1_loss(f_pred, f)
        elif self.target == 'aoe_c_only':
            loss = F.l1_loss(f_c_pred_x_gt, f)
        elif self.target == 'aoe_x_only':
            loss = F.l1_loss(f_c_gt_x_pred, f)
        elif self.target == 'f':
            loss = f_pred.mean()
        elif self.target == 'f_c_only':
            loss = f_c_pred_x_gt.mean()
        elif self.target == 'f_x_only':
            loss = f_c_gt_x_pred.mean()
        elif self.target == 'de':
            loss = F.mse_loss(x, x_pred)
        else:
            raise Exception('unknown target {self.target}')

        constr_violation = eq_constrs.abs().mean().item()
        if (ineq_constrs > 0).any():
            constr_violation += ineq_constrs[ineq_constrs > 0].mean().item()
        metrics = {
            'aoe': F.l1_loss(f_pred, f).item(),
            'obj': f_pred.mean().item(),
            'constr_volation': constr_violation,
        }

        return CMPState(loss=loss, ineq_defect=ineq_constrs, eq_defect=eq_constrs, misc=metrics)


def get_datasets(args: namedtuple) -> typing.Tuple[Subset, Subset, Subset]:
    dataset = SyntheticLPDataset("datasets/2var8cons_InvPLP0.tar.gz", "test", norm=True)

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
    dataset_args.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    dataset_args.add_argument('--slack', action='store_true', help='Slack is included in the dataset (model should use equality constraints, not inequalitiy constraints)')
    dataset_args.add_argument('--workers', type=int, default=2, help='Number of DataLoader workers')

    train_args = parser.add_argument_group('training', description='Training arguments')
    train_args.add_argument('--no_gpu', action='store_true', help='Do not use the GPU even if one is available')
    train_args.add_argument('--lr', type=float, default=1e-5, help='Optimizer learning rate')
    train_args.add_argument('--momentum', type=float, default=8e-2, help='Optimizer momentum')
    train_args.add_argument('--dual_restarts', action='store_true', help='Use dual restarts')
    train_args.add_argument('--no_extra_gradient', action='store_true', help='Use extra-gradient optimizers')
    train_args.add_argument('--max_epochs', type=int, default=500, help='Maximum number of training epochs')
    train_args.add_argument('--max_hours', type=int, default=3, help='Maximum hours to train')
    train_args.add_argument('--target', type=str, choices=['aoe', 'f', 'aoe_c_only', 'f_c_only', 'de'], default='aoe', help='Minimization target (aoe = l1 loss between predicted and true objective, f = true objective function)')
    train_args.add_argument('--nonneg_constr_method', type=str, choices=['none', 'softplus', 'relu', 'constr'], default='relu', help='How to ensure predicted decisions satisfy non-negativity constraint')

    model_args = parser.add_argument_group('logging', description='Logging arguments')
    model_args.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    model_args.add_argument('--wandb_exp', type=str, default=None, help='WandB experiment name')
    model_args.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='WandB tags')

    return parser

# this is a little random, but assemble the matrices into "Extended matrix representation"
# [0 c^T 0]
# [0 I   x]
# [1 -A  b]
def render_problem(A, b, c, x, f):
    first_row = torch.cat([torch.zeros(1), c, torch.zeros(1)]).unsqueeze(0)
    second_row = torch.cat([torch.zeros_like(x), torch.eye(x.size(0)), x], dim=1)
    third_row = torch.cat([torch.ones_like(b), -A, b], dim=1)

    img = torch.cat([first_row, second_row, third_row])
    return img

if __name__ == '__main__':
    args = get_argparser().parse_args()
    args.use_wandb = args.wandb_project is not None
    args.extra_gradient = not args.no_extra_gradient
    assert not (args.target == 'f' and args.nonneg_constr_method == 'none'), 'You must specify a method to enforce the non-negativity constraint if the minimization target is f'
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            config=args,
        )

    train_set = SyntheticLPDataset(args.dataset_path, "train", args.slack, norm=True)
    test_set = SyntheticLPDataset(args.dataset_path, "test", args.slack, norm=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    device = 'cuda:0' if torch.cuda.is_available() and not args.no_gpu else 'cpu'

    u, c, A, b, x, f = train_set[0]
    model = DeterministicLP(u.size(0), x.size(0), b.size(0))
    model.to(device)

    # problem specific stuff happens in here
    cmp = ForwardOptimization(train_set, device, args.target, args.nonneg_constr_method, 'eq' if args.slack else 'ineq')

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

    train_metrics = {
        'aoe': RunningAverage(output_transform=lambda x: x),
        'obj': RunningAverage(output_transform=lambda x: x),
        'constr_volation': RunningAverage(output_transform=lambda x: x),
        # 'nonneg_loss': RunningAverage(output_transform=lambda x: x)
    }
    val_metrics = {
        'sub_optimality': RunningAverage(output_transform=lambda x: x),
        'n_feasible': RunningAverage(output_transform=lambda x: x),
        'feasibility_volation': RunningAverage(output_transform=lambda x: x),
    }

    progress_bar = tqdm.trange(args.max_epochs)
    for epoch in progress_bar:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            lagrangian = formulation.composite_objective(cmp.closure, model, batch)

            formulation.custom_backward(lagrangian)
            if args.extra_gradient:
                optimizer.step(cmp.closure, model, batch)
            else:
                optimizer.step()

            for name, avg in train_metrics.items():
                avg.update(cmp.state.misc[name])

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = train_set.norm(*batch)

                (c, A, b, x, f), (c_pred, A_pred, b_pred, x_pred, f_pred) = cmp.sample(model, batch)
                # two eval metrics we care about
                # 1. is the predicted solution feasible wrt to the true parameters?
                obj = torch.bmm(c, x_pred).squeeze(2)
                # 2. how good is the predicted solution wrt to the true parameters?
                constr = torch.bmm(A, x_pred)
                if args.slack:
                    feas = torch.isclose(constr, b)
                else:
                    feas = constr < b

                constr_volation = (b[~feas] - constr[~feas])/b[~feas]
                all_feas = feas.all(dim=1)
                # sub_optimality = (f[all_feas] - obj[all_feas])/f[all_feas]
                sub_optimality = (f - obj)/f

                val_metrics['n_feasible'].update((feas.sum()/feas.numel()).item())
                val_metrics['feasibility_volation'].update(constr_volation.mean().item())
                val_metrics['sub_optimality'].update(sub_optimality.mean().item())

            #     cca = CCA()
            batch = tuple(x.unsqueeze(0) for x in test_set[0])
            batch = train_set.norm(*batch)
            (c_gt, A_gt, b_gt, x_gt, f_gt), (c_pred, A_pred, b_pred, x_pred, f_pred) = cmp.sample(model, batch)
            A_gt, b_gt, c_gt, x_gt, f_gt = A_gt.squeeze().cpu(), b_gt.squeeze(0).cpu(), c_gt.squeeze().cpu(), x_gt.squeeze(0).cpu(), f_gt.cpu()
            A_pred, b_pred, c_pred, x_pred, f_pred = A_pred.squeeze().cpu(), b_pred.squeeze(0).cpu(), c_pred.squeeze().cpu(), x_pred.squeeze(0).cpu(), f_pred.cpu()
            gt_img = render_problem(A_gt, b_gt, c_gt, x_gt, f_gt)
            pred_img = render_problem(A_pred, b_pred, c_pred, x_pred, f_pred)
            if args.use_wandb:
                img = wandb.Image(torch.cat([gt_img, pred_img], dim=1), caption='left: ground truth, right: prediction')
                wandb.log({"problem": img}, step=epoch)

        train_metrics_log = {name: avg.compute() for name, avg in train_metrics.items()}
        train_metrics_log['total_loss'] = train_metrics_log['constr_volation']
        if args.target == 'aoe_c_only' or args.target == 'aoe':
            train_metrics_log['total_loss'] += train_metrics_log['aoe']
        elif args.target == 'f_c_only' or args.target == 'f':
            train_metrics_log['total_loss'] += train_metrics_log['obj']

        val_metrics_log = {name: avg.compute() for name, avg in val_metrics.items()}

        progress_bar.set_postfix(dict(**train_metrics_log, **val_metrics_log))
        if args.use_wandb:
            wandb.log(train_metrics_log, step=epoch)

        for avg in train_metrics.values():
            avg.reset()

        for avg in val_metrics.values():
            avg.reset()