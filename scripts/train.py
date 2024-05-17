import argparse
import numpy as np
import pathlib
import sys
import torch
import torch.nn as nn
import logging

import xarray as xr

from collections import namedtuple
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver, Timer, TerminateOnNan, TimeLimit
from ignite.metrics.regression import R2Score, PearsonCorrelation
from torch.utils.data import DataLoader, Subset
from operator import itemgetter

from ..model import CVAE
from ..taxi.dataset import TaxiDataset

logger = logging.getLogger(__name__)

StepOutput = namedtuple('StepOutput', ['y', 'y_hat', 'z', 'z_hat'])

def get_datasets(args):
    dataset = TaxiDataset(args.dataroot, no_norm=args.no_norm, latent_cost=args.include_latent_cost, obs_slack=args.include_obs_slack)

    total_samples = len(dataset)
    indices = np.arange(total_samples).tolist() # dataloader shuffles, don't need to here
    split = np.array([args.train_prop, args.valid_prop, args.test_prop])
    assert np.isclose(split.sum(), 1), f'train-val-test split does not sum to 1, it sums to {split.sum()}'
    split_indices = np.cumsum(split*total_samples).astype(int)

    train_set = Subset(dataset, indices[0:split_indices[0]])
    val_set = Subset(dataset, indices[split_indices[0]:split_indices[1]])
    test_set = Subset(dataset, indices[split_indices[1]:split_indices[2]])

    return train_set, val_set, test_set

def train(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_set, val_set, _ = get_datasets(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, drop_last=True)

    obs, cond, _ = train_set[0]
    model = CVAE(obs.shape[0], action.shape[0], args.z_dim, args.hidden_dim)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=8e-3)

    def train_step(trainer, batch):
        model.train()
        optimizer.zero_grad()

        obs, cond, latents = batch
        obs, cond = obs.to(device), cond.to(device)

        prior, posterior, obs_hat, mse, kld = model(obs, cond)
        loss = args.kld_loss_weight*kld + (1 - args.kld_loss_weight)*mse
        loss.backward()
        optimizer.step()

        return StepOutput(obs.cpu(), obs_hat.cpu(), latents, prior.loc.cpu())

    trainer = Engine(train_step)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    if device != 'cpu':
        GpuInfo().attach(trainer, name='gpu')

    def eval_step(evaluator, batch):
        model.eval()
        with torch.no_grad():
            obs, cond, latents = batch
            obs, cond = obs.to(device), cond.to(device)

            prior, posterior, obs_hat, mse, kld = model(obs, cond)


    eval_metrics = {
        'R^2': R2Score(),
        'MCC': PearsonCorrelation()
    }
    create_supervised_evaluator(model, metrics=eval_metrics, model_transform=lambda model_output: model_output[0].loc)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(trainer):
        for key, value in trainer.state.output.items():
            print(f"Epoch[{trainer.state.epoch}] {key}: {value:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(trainer):
        pass

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(trainer):
        pass

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(trainer):
    #     evaluator.run(val_loader)
    #     metrics = evaluator.state.metrics
    #     print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    trainer.run(train_loader, max_epochs=args.epochs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=64, help='Max hidden dim for CVAE')
    parser.add_argument('--z_dim', type=int, default=10, help='Latent dim for CVAE')
    parser.add_argument('mode', choices=['cvae', 'ivae'], help='For compatibility')

    parser.add_argument("--dataset", type=str, required=True,
                        help="Type of the dataset to be used. 'toy-MANIFOLD/TRANSITION_MODEL'")
    parser.add_argument("--no_norm", action="store_true",
                        help="no normalization in toy datasets")
    parser.add_argument("--dataroot", type=str, default="./",
                        help="path to dataset")
    parser.add_argument("--include_latent_cost", action="store_true",
                        help="taxi dataset: include cost parameters in latents")
    parser.add_argument("--include_obs_slack", action="store_true",
                        help="taxi dataset: include unused capacity and unserved demand in observation")
    parser.add_argument("--train_prop", type=float, default=None,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--valid_prop", type=float, default=0.10,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--test_prop", type=float, default=0.10,
                        help="proportion of all samples used in test set")
    parser.add_argument("--n_workers", type=int, default=2,
                        help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="batch size used during training")
    parser.add_argument("--eval_batch_size", type=int, default=1024,
                        help="batch size used during evaluation")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of epochs to train for")
    parser.add_argument("--time_limit", type=float, default=None,
                        help="After this amount of time, terminate training.")

    parser.add_argument("--output_dir", required=True,
                        help="Directory to output logs and model checkpoints")
    parser.add_argument("--fresh", action="store_true",
                        help="Remove output directory before starting, even if experiment is completed.")
    parser.add_argument("--ckpt_period", type=int, default=50000,
                        help="Number of batch iterations between each checkpoint.")
    parser.add_argument("--eval_period", type=int, default=5000,
                        help="Number of batch iterations between each evaluation on the validation set.")
    parser.add_argument("--fast_log_period", type=int, default=100,
                        help="Number of batch iterations between each cheap log.")
    parser.add_argument("--plot_period", type=int, default=10000,
                        help="Number of batch iterations between each cheap log.")
    parser.add_argument("--scheduler", type=str, default="reduce_on_plateau", choices=["reduce_on_plateau"],
                        help="Patience for reducing the learning rate in terms of evaluations on tye validation set")
    parser.add_argument("--scheduler_patience", type=int, default=120,
                        help="(applies only to reduce_on_plateau) Patience for reducing the learning rate in terms of evaluations on tye validation set")
    parser.add_argument("--best_criterion", type=str, default="loss", choices=["loss", "nll"],
                        help="Criterion to look at for saving best model and early stopping. loss include regularization terms")
    parser.add_argument('--no_print', action="store_true",
                        help='do not print')
    parser.add_argument('--comet_key', type=str, default=None,
                        help="comet api-key")
    parser.add_argument('--comet_tag', type=str, default=None,
                        help="comet tag, to ease comparison")
    parser.add_argument('--comet_workspace', type=str, default=None,
                        help="comet workspace")
    parser.add_argument('--comet_project_name', type=str, default=None,
                        help="comet project_name")

    args = parser.parse_args()
    train(args)
