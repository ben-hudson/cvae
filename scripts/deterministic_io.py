import argparse
import cooper
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
import tqdm
import multiprocessing as mp

from cooper import ConstrainedMinimizationProblem as CMP, CMPState
from torch.utils.data import DataLoader

from synthetic_lp.dataset import SyntheticLPDataset

from ignite.metrics import RunningAverage
from torchvision.utils import make_grid
from models.deterministic_lp import DeterministicLP
from scipy.optimize import linprog

def solve_batch(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, workers=None):
    c = c.cpu()
    A_ub = A_ub.cpu() if A_ub is not None else [None]*len(c)
    b_ub = b_ub.cpu() if b_ub is not None else [None]*len(c)
    A_eq = A_eq.cpu() if A_eq is not None else [None]*len(c)
    b_eq = b_eq.cpu() if b_eq is not None else [None]*len(c)
    bounds = [(None, None)]*len(c) # x does not have non-negativity constraints in the synthetic LP dataset

    with mp.Pool(processes=workers) as workers:
        results = workers.starmap(linprog, zip(c, A_ub, b_ub, A_eq, b_eq, bounds))

    f, x, success = zip(*[(result.fun, result.x, result.success) for result in results if result.success])
    # Creating a tensor from a list of numpy.ndarrays is extremely slow.
    # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
    f = np.array(f)
    x = np.array(x)
    success = np.array(success)
    return torch.Tensor(f).unsqueeze(1), torch.Tensor(x).unsqueeze(2), torch.Tensor(success)


def get_val_metrics(c, A, b, x_pred, f, constr_type):
    # there are three metrics we care about
    # 1. is the predicted solution feasible wrt to the true problem?
    # 2. if not, how infeasible is it?
    # 3. how optimal is it?

    obj = torch.bmm(c, x_pred).squeeze(2)
    constr = torch.bmm(A, x_pred)
    if constr_type == 'eq':
        feas = torch.isclose(constr, b)
    else:
        feas = constr < b
    constr_volation = (b[~feas] - constr[~feas])/b[~feas]
    constr_volation = constr[~feas]/b[~feas] - 1
    sub_optimality = obj/f - 1

    return {
        'other/feas_frac': feas.sum()/feas.numel(), # fraction of satisfied constraints
        'other/rel_feas_violation': constr_volation.mean(),
        'other/rel_sub_optimality': sub_optimality.mean(),
    }

class ForwardOptimization(CMP):
    def __init__(self, unnorm_fun, device, loss, constr, extra_constrs):
        super().__init__(is_constrained=True)

        self.unnorm = unnorm_fun
        self.device = device

        self.loss = loss
        self.constr = constr
        self.extra_constrs = extra_constrs

    def sample(self, model, batch):
        u_normed, c_normed, A_normed, b_normed, x_normed, f_normed = batch
        _, c, A, b, x, f = self.unnorm(u_normed, c_normed, A_normed, b_normed, x_normed, f_normed)
        c = c.to(self.device).unsqueeze(1)
        A = A.to(self.device)
        b = b.to(self.device).unsqueeze(2)
        x = x.to(self.device).unsqueeze(2)
        f = f.to(self.device)

        u_normed = u_normed.to(self.device)
        c_pred_normed, A_pred_normed, b_pred_normed, x_pred_normed = model(u_normed)

        # pass zeros for u and f as placeholders
        _, c_pred, A_pred, b_pred, x_pred, _ = self.unnorm(torch.zeros_like(u_normed), c_pred_normed, A_pred_normed, b_pred_normed, x_pred_normed, torch.zeros_like(f_normed))
        x_pred = x_pred.unsqueeze(2)
        c_pred = c_pred.unsqueeze(1)
        b_pred = b_pred.unsqueeze(2)
        f_pred = torch.bmm(c_pred, x_pred).squeeze(2)

        return (c, A, b, x, f), (c_pred, A_pred, b_pred, x_pred, f_pred)

    def closure(self, model, batch):
        (c, A, b, x, f), (c_pred, A_pred, b_pred, x_pred, f_pred) = self.sample(model, batch)

        # we observe x and f, and we want to use them to get losses on all of the problem parameters
        # there are a lot of constraints we can impose, the question is which ones?
        # A_pred * x_pred <= (or =) b_pred
        # A_pred * x <= (or =) b_pred
        # f == c_pred*x_pred
        # f == c_pred*x
        # for now, we calculate all of them and report them as metrics
        x_pred_constr = torch.bmm(A_pred, x_pred) - b_pred
        x_true_constr = torch.bmm(A_pred, x) - b_pred
        x_pred_obj = f_pred
        x_true_obj = torch.bmm(c_pred, x).squeeze(2)
        x_pred_obj_err = x_pred_obj - f
        x_true_obj_err = x_true_obj - f

        # there are also a lot of losses
        # some of of the losses described in https://arxiv.org/abs/2109.03920
        md_loss = F.mse_loss(x_pred, x)
        aoe_loss = x_pred_obj_err.abs().mean()
        roe_loss = torch.abs(f_pred/f - 1).mean() # TODO: implement the scaling invariance constraint here
        f_loss = x_pred_obj.mean() # equivalent to minimizing violation of kkt conditions when we are doing constraint optimization already

        # pick a loss
        if self.loss == 'md':
            loss = md_loss

        elif self.loss == 'aoe' or self.loss == 'x_pred_obj_err':
            assert 'x_pred_obj' not in self.extra_constrs, "can't minimize |f - f^| and constrain f = f^"
            loss = x_pred_obj_err.abs().mean()

        elif self.loss == 'x_true_obj_err':
            assert 'x_true_obj' not in self.extra_constrs, "can't minimize |f - c^ * x| and constrain f = c^ * x"
            loss = x_true_obj_err.abs().mean()

        elif self.loss == 'roe':
            loss = roe_loss

        elif self.loss == 'f' or self.loss == 'x_pred_obj':
            assert 'x_pred_obj' not in self.extra_constrs, "can't minimize f^ and constrain f = f^"
            loss = x_pred_obj.mean()

        elif self.loss == 'f' or self.loss == 'x_true_obj':
            assert 'x_true_obj' not in self.extra_constrs, "can't minimize c^ * x and constrain f = c^ * x"
            loss = x_true_obj.mean()

        else:
            raise ValueError(f'unknown loss {self.loss}')

        # set up constraints
        ineq_constrs = []
        eq_constrs = []
        if self.constr == 'ineq':
            if 'x_pred_constr' in self.extra_constrs:
                ineq_constrs.append(x_pred_constr)
            if 'x_true_constr' in self.extra_constrs:
                ineq_constrs.append(x_true_constr)
        elif self.constr == 'eq':
            if 'x_pred_constr' in self.extra_constrs:
                eq_constrs.append(x_pred_constr)
            if 'x_true_constr' in self.extra_constrs:
                eq_constrs.append(x_true_constr)
        else:
            raise ValueError(f'unknown constraint {self.constr}')

        if 'x_pred_obj' in self.extra_constrs:
            eq_constrs.append(x_pred_obj)
        if 'x_true_obj' in self.extra_constrs:
            eq_constrs.append(x_true_obj)

        ineq_defect = None if len(ineq_constrs) == 0 else torch.cat(ineq_constrs, dim=1)
        eq_defect = None if len(eq_constrs) == 0 else torch.cat(eq_constrs, dim=1)

        # finally, some aggregated metrics
        metrics = {
            # the losses
            'loss/md_loss': md_loss.detach(),
            'loss/aoe_loss': aoe_loss.detach(),
            'loss/roe_loss': roe_loss.detach(),
            'loss/f_loss': f_loss.detach(),

            # the constraint violations
            # inequality constraint values are positive when violated
            'constr/x_pred_constr_loss': x_pred_constr.detach().clamp(min=0).mean() if self.constr == 'ineq' else x_pred_constr.detach().abs().mean(),
            'constr/x_true_constr_loss': x_true_constr.detach().clamp(min=0).mean() if self.constr == 'ineq' else x_true_constr.detach().abs().mean(),
            'constr/x_pred_obj_loss': x_pred_obj.detach().abs().mean(),
            'constr/x_true_obj_loss': x_true_obj.detach().abs().mean(),

            # and the validation metrics for the training data
            **get_val_metrics(c, A, b, x_pred, f, self.constr)
        }

        return CMPState(loss=loss, ineq_defect=ineq_defect, eq_defect=eq_defect, misc=metrics)

def get_argparser():
    parser = argparse.ArgumentParser('Train an MLP to approximate an LP solver using constrained optimization')

    dataset_args = parser.add_argument_group('dataset', description='Dataset arguments')
    dataset_args.add_argument('dataset_path', type=pathlib.Path, help='Path to dataset')
    dataset_args.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    dataset_args.add_argument('--constr', choices=['ineq', 'eq'], default='ineq', help='Data represents a problem with inequality or equality constraints')
    dataset_args.add_argument('--workers', type=int, default=2, help='Number of DataLoader workers')

    train_args = parser.add_argument_group('training', description='Training arguments')
    train_args.add_argument('--no_gpu', action='store_true', help='Do not use the GPU even if one is available')
    train_args.add_argument('--lr', type=float, default=1e-5, help='Optimizer learning rate')
    train_args.add_argument('--momentum', type=float, default=8e-2, help='Optimizer momentum')
    train_args.add_argument('--dual_restarts', action='store_true', help='Use dual restarts')
    train_args.add_argument('--no_extra_gradient', action='store_true', help='Use extra-gradient optimizers')
    train_args.add_argument('--max_epochs', type=int, default=500, help='Maximum number of training epochs')
    train_args.add_argument('--max_hours', type=int, default=3, help='Maximum hours to train')
    train_args.add_argument('--loss', type=str, choices=['md', 'aoe', 'roe', 'f', 'x_pred_obj_err', 'x_true_obj_err', 'x_pred_obj', 'x_true_obj'], default='aoe', help='Minimization target')
    train_args.add_argument('--extra_constrs', type=str, nargs='*', choices=['x_pred_constr', 'x_true_constr', 'x_pred_obj', 'x_true_obj'], default=['x_pred_constr'], help='Constraints to add')
    train_args.add_argument('--solve_exact', action='store_true', help='Solve the problems exactly during validation')

    model_args = parser.add_argument_group('logging', description='Logging arguments')
    model_args.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    model_args.add_argument('--wandb_exp', type=str, default=None, help='WandB experiment name')
    model_args.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='WandB tags')

    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    args.extra_gradient = not args.no_extra_gradient
    args.use_wandb = args.wandb_project is not None
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            config=args,
        )

    train_set = SyntheticLPDataset(args.dataset_path, "train", args.constr == "eq", norm=True)
    test_set = SyntheticLPDataset(args.dataset_path, "test", args.constr == "eq", norm=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=min(args.batch_size, len(test_set)), shuffle=False, num_workers=args.workers, drop_last=True)

    device = 'cuda:0' if torch.cuda.is_available() and not args.no_gpu else 'cpu'

    u, c, A, b, x, f = train_set[0]
    model = DeterministicLP(u.size(0), x.size(0), b.size(0))
    model.to(device)

    # training logic happens here
    cmp = ForwardOptimization(train_set.unnorm, device, args.loss, args.constr, args.extra_constrs)

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

    # these get populated automatically
    metrics = {}

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

            for name, value in cmp.state.misc.items():
                name = 'train/' + name
                if name not in metrics:
                    metrics[name] = RunningAverage(output_transform=lambda x: x)
                metrics[name].update(value)

        model.eval()
        with torch.no_grad():

            imgs = []

            for batch in test_loader:
                batch = train_set.norm(*batch)

                (c, A, b, x, f), (c_pred, A_pred, b_pred, x_pred, f_pred) = cmp.sample(model, batch)

                if args.solve_exact:
                    if args.constr == 'eq':
                        f_pred, x_pred, solve_success = solve_batch(c_pred, A_eq=A_pred, b_eq=b_pred, workers=args.workers)
                        x_pred = x_pred.to(device)
                    elif args.constr == 'ineq':
                        f_pred, x_pred, solve_success = solve_batch(c_pred, A_ub=A_pred, b_ub=b_pred, workers=args.workers)
                        x_pred = x_pred.to(device)

                for name, value in get_val_metrics(c, A, b, x_pred, f, args.constr).items():
                    name = 'val/' + name
                    if name not in metrics:
                        metrics[name] = RunningAverage(output_transform=lambda x: x)
                    metrics[name].update(value)

                # render the first problem in each batch
                true_problem = SyntheticLPDataset.render(None, c[0], A[0], b[0], x[0], f[0])
                pred_problem = SyntheticLPDataset.render(None, c_pred[0], A_pred[0], b_pred[0], x_pred[0], f_pred[0])
                imgs.append(torch.cat([true_problem, pred_problem], dim=1))

            if args.use_wandb:
                img = wandb.Image(make_grid(imgs), caption='left: ground truth, right: prediction')
                wandb.log({"examples": img}, step=epoch)

        if args.use_wandb:
            wandb.log({name: avg.compute() for name, avg in metrics.items()}, step=epoch)

        for avg in metrics.values():
            avg.reset()
