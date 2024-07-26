import pyepo
import argparse
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
import tqdm
import multiprocessing as mp

from torch.utils.data import DataLoader, Subset
from torch.distributions import kl_divergence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import linear_sum_assignment

from ignite.metrics import RunningAverage
from torchvision.utils import make_grid
from cvae import CVAE

def get_loss(prior, posterior, obs_hat, obs):
    kld = kl_divergence(prior, posterior).sum()
    bce = F.binary_cross_entropy_with_logits(obs_hat, obs, reduction="sum")
    loss = kld + bce
    metrics = {
        "kld": kld.detach(),
        "bce": bce.detach(),
        "loss": loss.detach()
    }
    return loss, metrics

def get_metrics(latents_hat, latents, obs_hat, obs):
    latents_hat_np = latents_hat.cpu().numpy()
    latents_np = latents.cpu().numpy()

    # mean correlation coefficient
    # from https://github.com/ilkhem/icebeem/blob/0077f0120c83bcc6d9b199b831485c42bed2401f/metrics/mcc.py#L391
    d = latents_hat_np.shape[1]
    cc = np.corrcoef(latents_hat_np, latents_np, rowvar=False, dtype=np.float32)
    cc = np.abs(cc[:d, d:]) # remove self-correlations
    mcc = cc[linear_sum_assignment(-1 * cc)].mean()

    return {
        "mcc": mcc
    }


def get_argparser():
    parser = argparse.ArgumentParser('Train an MLP to approximate an LP solver using constrained optimization')

    dataset_args = parser.add_argument_group('dataset', description='Dataset arguments')
    dataset_args.add_argument('dataset', type=str, choices=['shortestpath', 'knapsack', 'tsp', 'portfolio'], help='Dataset to generate')
    dataset_args.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    dataset_args.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    dataset_args.add_argument('--workers', type=int, default=2, help='Number of DataLoader workers')

    train_args = parser.add_argument_group('training', description='Training arguments')
    train_args.add_argument('--no_gpu', action='store_true', help='Do not use the GPU even if one is available')
    train_args.add_argument('--lr', type=float, default=1e-5, help='Optimizer learning rate')
    train_args.add_argument('--momentum', type=float, default=8e-2, help='Optimizer momentum')
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

    if args.dataset == 'shortestpath':
        grid = (5,5) # grid size
        model = pyepo.model.grb.shortestPathModel(grid)

        # generate data
        num_data = args.samples # number of data
        num_feat = 5 # size of feature
        deg = 4 # polynomial degree
        noise_width = 0.5 # noise width
        context, costs = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

        # build dataset
        dataset = pyepo.data.dataset.optDataset(model, context, costs)
    else:
        raise ValueError('NYI')

    indices = torch.randperm(len(dataset))
    train_indices, test_indices = train_test_split(indices, test_size=0.2)
    train_set, test_set = Subset(dataset, train_indices), Subset(dataset, test_indices)
    train_loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=min(args.batch_size, len(test_set)), shuffle=False, num_workers=args.workers, drop_last=True)

    # we need to scale the generated data and solutions
    context_scaler, obs_scaler = StandardScaler(), MinMaxScaler()
    for context, costs, sols, objs in train_loader:
        context_scaler.partial_fit(context)

        # obs = torch.cat([context, sols, objs], dim=-1)
        obs = sols
        obs_scaler.partial_fit(obs)

    device = 'cuda:0' if torch.cuda.is_available() and not args.no_gpu else 'cpu'

    latent_dim = 40
    latent_type = "normal"
    model = CVAE(context.size(-1), obs.size(-1), latent_dim, latent_type)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # these get populated automatically
    metrics = {}

    progress_bar = tqdm.trange(args.max_epochs)
    for epoch in progress_bar:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            # context, costs, sols, objs = batch

            context = context_scaler.transform(batch[0])
            context = torch.Tensor(context).to(device)

            # obs = np.hstack((batch[0], batch[2], batch[3]))
            obs = obs_scaler.transform(batch[2])
            obs = torch.Tensor(obs).to(device) # obs is binary

            latents = batch[1].to(device)

            prior, posterior, latents_hat, obs_hat = model(context, obs)

            loss, loss_metrics = get_loss(prior, posterior, obs_hat, obs)
            loss.backward()

            optimizer.step()

            for name, value in loss_metrics.items():
                name = 'train/' + name
                if name not in metrics:
                    metrics[name] = RunningAverage(output_transform=lambda x: x)
                metrics[name].update(value)

        model.eval()
        with torch.no_grad():

            imgs = []

            for batch in test_loader:

                context = context_scaler.transform(batch[0])
                context = torch.Tensor(context).to(device)

                obs = obs_scaler.transform(batch[2])
                obs = torch.Tensor(obs).to(device)

                latents = batch[1].to(device)

                prior, posterior, latents_hat, obs_hat = model(context, obs)

                loss, loss_metrics = get_loss(prior, posterior, obs_hat, obs)
                other_metrics = get_metrics(latents_hat, latents, obs_hat, obs)

                for name, value in dict(**loss_metrics, **other_metrics).items():
                    name = 'val/' + name
                    if name not in metrics:
                        metrics[name] = RunningAverage(output_transform=lambda x: x)
                    metrics[name].update(value)

        if args.use_wandb:
            wandb.log({name: avg.compute() for name, avg in metrics.items()}, step=epoch)

        for avg in metrics.values():
            avg.reset()
