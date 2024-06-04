import argparse
import numpy as np
import torch
import torch.nn.functional as F

from collections import namedtuple
from models.cvae import CVAE
from models.solver_vae import SolverVAE
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan, TimeLimit, ProgressBar
from ignite.metrics import RunningAverage
from taxi.allocation_env import AllocationDataset
from torch.utils.data import DataLoader, Subset
from torch.distributions import kl_divergence
from typing import Tuple, List, Union
from itertools import pairwise
from sklearn.preprocessing import StandardScaler

TrainState = namedtuple('TrainState', ['kld', 'mse', 'loss'])
EvalState = namedtuple('EvalState', ['y', 'y_hat', 'q', 'q_hat', 'W', 'W_hat', 'h', 'h_hat'])

def get_datasets(args: namedtuple) -> Tuple[Subset, Subset, Subset]:
    dataset = AllocationDataset(args.n_nodes, args.n_producers, args.n_consumers, args.samples)

    samples = len(dataset)
    indices = np.arange(samples).tolist() # dataloader shuffles, don't need to here
    split = np.array([0, args.train_frac, args.val_frac, args.test_frac])
    assert np.isclose(split.sum(), 1), f'train-val-test split does not sum to 1, it sums to {split.sum()}'

    split_indices = np.cumsum(split*samples).astype(int)
    subsets = tuple(Subset(dataset, indices[start:end]) for start, end in pairwise(split_indices))
    return subsets

def get_trainer(model: torch.nn.Module, args: namedtuple, device: Union[torch.device, str]='cpu') -> Tuple[Engine, torch.optim.Optimizer]:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    assert args.kld_weight >= 0 and args.kld_weight <= 1, f'kld weight must be between 0 and 1, it is {args.kld_weight}'
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y, W, h, q, Q = batch

        x = x.float().to(device)
        y = y.float().to(device)
        Q = Q.float().to(device)

        priors, posteriors, sample = model(y, x)
        Q_sample = sample[5]

        # "absolute objective error" loss from https://arxiv.org/abs/2006.08923
        mse = F.l1_loss(Q_sample, Q).sum()

        # want posterior to be close to the prior
        kld = torch.zeros(1, device=device)
        for p, q in zip(priors, posteriors):
            kld += kl_divergence(p, q).sum()

        loss = args.kld_weight*kld + (1 - args.kld_weight)*mse
        loss.backward()
        optimizer.step()

        return TrainState(kld.cpu().item(), mse.cpu().item(), loss.cpu().item())

    trainer = Engine(train_step)
    RunningAverage(output_transform=lambda state: state.kld).attach(trainer, 'KLD')
    RunningAverage(output_transform=lambda state: state.mse).attach(trainer, 'MSE')
    RunningAverage(output_transform=lambda state: state.loss).attach(trainer, 'Loss')
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TimeLimit(limit_sec=args.max_hours*3600))

    return trainer, optimizer

def get_evaluator(model: torch.nn.Module, args: namedtuple, device='cpu'):
    def eval_step(evaluator, batch):
        model.eval()
        with torch.no_grad():
            x, y, W, h, q, Q = batch

            y = y.float().to(device)
            x = x.float().to(device)

            _, _, (_, y_hat, W_hat, h_hat, q_hat, _) = model(y, x)

            return EvalState(y.cpu(), y_hat.cpu(), q, q_hat.cpu(), W, W_hat.cpu(), h, h_hat.cpu())

    evaluator = Engine(eval_step)
    return evaluator

def setup_wandb_logger(args: namedtuple, trainer: Engine, evaluator: Engine, optimizer: torch.optim.Optimizer):
    from ignite.handlers.wandb_logger import WandBLogger

    wandb_logger = WandBLogger(
        project=args.wandb_project,
        name=args.wandb_exp,
        config=args,
        tags=args.wandb_tags
    )

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag='Training',
        metric_names='all'
    )

    wandb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag='Validation',
        metric_names='all',
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    return wandb_logger

def log_latents_to_wandb(wandb, state: EvalState) -> None:
    images = dict()

    y = state.output.y.mean(dim=0).unsqueeze(0)
    y_hat = state.output.y_hat.mean(dim=0).unsqueeze(0)
    images['y'] = wandb.Image(torch.cat([y, y_hat], dim=0))

    W = state.output.W.mean(dim=0)
    W_hat = state.output.W_hat.mean(dim=0)
    images['W'] = wandb.Image(torch.cat([W, W_hat], dim=1))

    q = state.output.q.mean(dim=0).unsqueeze(0)
    q_hat = state.output.q_hat.mean(dim=0).unsqueeze(0)
    images['q'] = wandb.Image(torch.cat([q, q_hat], dim=0))

    h = state.output.h.mean(dim=0).unsqueeze(0)
    h_hat = state.output.h_hat.mean(dim=0).unsqueeze(0)
    images['h'] = wandb.Image(torch.cat([h, h_hat], dim=0))

    wandb.log(images)

def get_argparser():
    parser = argparse.ArgumentParser('Train CVAE on a dataset')

    env_args = parser.add_argument_group('env', description='Environment arguments')
    env_args.add_argument('--n_nodes', type=int, default=4, help='Number of nodes in the network')
    env_args.add_argument('--n_consumers', type=int, default=4, help='Number of producers in the allocation problem')
    env_args.add_argument('--n_producers', type=int, default=4, help='Number of consumers in the allocation problem')

    dataset_args = parser.add_argument_group('dataset', description='Dataset arguments')
    dataset_args.add_argument('--samples', type=int, default=10 ** 6, help='Number of samples to draw from the environment')
    dataset_args.add_argument('--train_frac', type=float, default=0.8, help='Fraction of dataset used for training')
    dataset_args.add_argument('--val_frac', type=float, default=0.1, help='Fraction of dataset used for validation')
    dataset_args.add_argument('--test_frac', type=float, default=0.1, help='Fraction of dataset used for testing')
    dataset_args.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    dataset_args.add_argument('--workers', type=int, default=2, help='Number of DataLoader workers')

    train_args = parser.add_argument_group('training', description='Training arguments')
    train_args.add_argument('--no_gpu', action='store_true', help='Do not use the GPU even if one is available')
    train_args.add_argument('--lr', type=float, default=1e-4, help='Optimizer learning rate')
    train_args.add_argument('--momentum', type=float, default=8e-3, help='Optimizer momentum')
    train_args.add_argument('--kld_weight', type=float, default=0.5, help='Relative weighting of KLD and MSE in training objective')
    train_args.add_argument('--max_epochs', type=int, default=500, help='Maximum number of training epochs')
    train_args.add_argument('--max_hours', type=int, default=3, help='Maximum hours to train')

    model_args = parser.add_argument_group('model', description='Model arguments')
    model_args.add_argument('--model', type=str, choices=['cvae', 'solver_vae'], default='cvae', help='Model to use')
    model_args.add_argument('--latent_dim', type=int, default=10, help='Model latent dimension')
    model_args.add_argument('--latent_dim_to_gt', action='store_true', help='Set model latent dimension to ground truth')
    model_args.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension in model MLPs')
    model_args.add_argument('--solver', type=str, choices=['cvx', 'lin'], default='cvx', help='Solver to use for forward problem')

    model_args = parser.add_argument_group('logging', description='Logging arguments')
    model_args.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    model_args.add_argument('--wandb_exp', type=str, default=None, help='WandB experiment name')
    model_args.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='WandB tags')

    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    args.use_wandb = args.wandb_project is not None

    train_set, val_set, test_set = get_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    device = 'cuda:0' if torch.cuda.is_available() and not args.no_gpu else 'cpu'

    x, y, _, h, _, _ = batch = train_set[0]
    if args.model == 'cvae':
        model = CVAE(
            y.shape[0],
            x.shape[0],
            h.shape[0] if args.latent_dim_to_gt else args.latent_dim,
            args.hidden_dim
        )
    elif args.model == 'solver_vae':
        model = SolverVAE(
            y.shape[0],
            x.shape[0],
            h.shape[0] if args.latent_dim_to_gt else args.latent_dim,
            args.hidden_dim,
            args.solver
        )
    else:
        raise ValueError(f'unknown model {args.model}')
    model.to(device)

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    trainer, optimizer = get_trainer(model, args, device=device)
    evaluator = get_evaluator(model, args, device=device)

    progress = ProgressBar(persist=True)
    progress.attach(trainer, event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED, metric_names='all')

    if args.use_wandb:
        wandb_logger = setup_wandb_logger(args, trainer, evaluator, optimizer)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_validation_metrics(trainer: Engine):
        evaluator.run(val_loader)
        if args.use_wandb:
            log_latents_to_wandb(wandb_logger._wandb, evaluator.state)

    trainer.run(train_loader, max_epochs=args.max_epochs)