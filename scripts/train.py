import argparse
import numpy as np
import pathlib
import torch

from collections import namedtuple
from models.cvae import CVAE
from models.solver_vae import SolverVAE
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan, TimeLimit, ProgressBar
from ignite.metrics import RunningAverage
from metrics import mean_corr_coef_np as mcc_score, r2_score
from taxi.dataset import TaxiDataset
from torch.utils.data import DataLoader, Subset
from typing import Tuple

TrainState = namedtuple('TrainState', ['kld', 'mse', 'loss'])
EvalState = namedtuple('EvalState', ['r2', 'mcc'])

# pairwise is available in itertools in Python 3.10+
# https://docs.python.org/3/library/itertools.html#itertools.pairwise
def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b

def get_datasets(args):
    dataset = TaxiDataset(args.path, no_norm=args.no_norm, latent_cost=args.latent_cost, obs_slack=args.obs_slack)

    total_samples = len(dataset)
    indices = np.arange(total_samples).tolist() # dataloader shuffles, don't need to here
    split = np.array([0, args.train_frac, args.val_frac, args.test_frac])
    assert np.isclose(split.sum(), 1), f'train-val-test split does not sum to 1, it sums to {split.sum()}'

    split_indices = np.cumsum(split*total_samples).astype(int)
    subsets = tuple(Subset(dataset, indices[start:end]) for start, end in pairwise(split_indices))
    return subsets

def get_trainer(model: torch.nn.Module, args: namedtuple, device='cpu') -> Tuple[Engine, torch.optim.Optimizer]:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    assert args.kld_weight >= 0 and args.kld_weight <= 1, f'kld weight must be between 0 and 1, it is {args.kld_weight}'
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        obs, cond, latent = batch
        obs, cond = obs.to(device), cond.to(device)

        prior, posterior, obs_hat, mse, kld = model(obs, cond)
        loss = args.kld_weight*kld + (1 - args.kld_weight)*mse
        loss.backward()
        optimizer.step()

        return TrainState(kld.cpu().item(), mse.cpu().item(), loss.cpu().item())

    trainer = Engine(train_step)
    RunningAverage(output_transform=lambda state: state.kld).attach(trainer, 'KLD')
    RunningAverage(output_transform=lambda state: state.mse).attach(trainer, 'MSE')
    RunningAverage(output_transform=lambda state: state.loss).attach(trainer, 'Loss')
    if device != 'cpu':
        GpuInfo().attach(trainer, name='gpu')
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TimeLimit(limit_sec=args.max_hours*3600))

    return trainer, optimizer

def get_evaluator(model: torch.nn.Module, args: namedtuple, device='cpu'):
    def eval_step(evaluator: Engine, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        model.eval()
        with torch.no_grad():
            obs, cond, latent = batch
            obs, cond = obs.to(device), cond.to(device)

            prior, obs_hat = model.sample(cond)
            prior_mean = prior.loc.cpu()

            r2 = r2_score(latent, prior_mean)
            mcc = mcc_score(latent, prior_mean)

            return EvalState(r2, mcc)

    evaluator = Engine(eval_step)
    RunningAverage(output_transform=lambda state: state.r2).attach(evaluator, 'R^2')
    RunningAverage(output_transform=lambda state: state.mcc).attach(evaluator, 'MCC')
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

    # wandb_logger.attach_opt_params_handler(
    #     trainer,
    #     event_name=Events.ITERATION_STARTED,
    #     optimizer=optimizer,
    # )

    wandb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag='Validation',
        metric_names='all',
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    return wandb_logger

def get_argparser():
    parser = argparse.ArgumentParser('Train CVAE on a dataset')
    dataset_args = parser.add_argument_group('dataset', description='Dataset arguments')
    dataset_args.add_argument('path', type=pathlib.Path, help='Path to dataset')
    dataset_args.add_argument('--no_norm', action='store_true', help='Do not convert the dataset to unit variance')
    dataset_args.add_argument('--latent_cost', action='store_true', help='Include cost coefficients q in the ground truth latents')
    dataset_args.add_argument('--obs_slack', action='store_true', help='Include slack variables in the observations')
    dataset_args.add_argument('--train_frac', type=float, default=0.8, help='Fraction of dataset used for training')
    dataset_args.add_argument('--val_frac', type=float, default=0.1, help='Fraction of dataset used for validation')
    dataset_args.add_argument('--test_frac', type=float, default=0.1, help='Fraction of dataset used for testing')
    dataset_args.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    dataset_args.add_argument('--workers', type=int, default=2, help='Number of DataLoader workers')

    train_args = parser.add_argument_group('training', description='Training arguments')
    train_args.add_argument('--no_gpu', action='store_true', help='Do not use the GPU even if one is available')
    train_args.add_argument('--lr', type=float, default=5e-4, help='Optimizer learning rate')
    train_args.add_argument('--momentum', type=float, default=8e-3, help='Optimizer momentum')
    train_args.add_argument('--kld_weight', type=float, default=0.5, help='Relative weighting of KLD and MSE in training objective')
    train_args.add_argument('--max_epochs', type=int, default=500, help='Maximum number of training epochs')
    train_args.add_argument('--max_hours', type=int, default=3, help='Maximum hours to train')

    model_args = parser.add_argument_group('model', description='Model arguments')
    model_args.add_argument('--model', type=str, choices=['cvae', 'solver_vae'], default='cvae', help='Model to use')
    model_args.add_argument('--latent_dim', type=int, default=10, help='Model latent dimension')
    model_args.add_argument('--latent_dim_to_gt', action='store_true', help='Set model latent dimension to ground truth')
    model_args.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension in model MLPs')

    model_args = parser.add_argument_group('logging', description='Logging arguments')
    model_args.add_argument('--wandb_project', type=str, default=None, help='WandB project name')
    model_args.add_argument('--wandb_exp', type=str, default=None, help='WandB experiment name')
    model_args.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='WandB tags')

    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()

    train_set, val_set, test_set = get_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    device = 'cuda:0' if torch.cuda.is_available() and not args.no_gpu else 'cpu'

    obs, cond, latent = train_set[0]
    if args.model == 'cvae':
        model = CVAE(
            obs.shape[0],
            cond.shape[0],
            latent.shape[0] if args.latent_dim_to_gt else args.latent_dim,
            args.hidden_dim
        )
    elif args.model == 'solver_vae':
        model = SolverVAE(
            obs.shape[0],
            cond.shape[0],
            latent.shape[0] if args.latent_dim_to_gt else args.latent_dim,
            args.hidden_dim
        )
    else:
        raise ValueError(f'unknown model {args.model}')
    model.to(device)

    trainer, optimizer = get_trainer(model, args, device=device)
    evaluator = get_evaluator(model, args, device=device)

    progress = ProgressBar(persist=True)
    progress.attach(trainer, event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED)

    if args.wandb_project is not None:
        setup_wandb_logger(args, trainer, evaluator, optimizer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(trainer: Engine):
        evaluator.run(val_loader)
        # hacky, but the only way to get trainer and evaluator metrics on the same bar
        metrics = dict(**trainer.state.metrics, **evaluator.state.metrics)
        progress.pbar.set_postfix(metrics)

    trainer.run(train_loader, max_epochs=args.max_epochs)