import numpy as np
import torch
import torch.utils
import xarray as xr

class TaxiDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, no_norm=False, latent_cost=False, obs_slack=False):
        self.data = xr.open_dataset(datapath)

        self.norm = not no_norm

        capacity = self.data.capacity.values
        demand = self.data.demand.values
        if latent_cost:
            triu_rows, triu_cols = np.triu_indices(self.data.cost.shape[1], k=1)
            cost = self.data.cost.values[:, triu_rows, triu_cols] # select half of cost matrix because it is symmetric
            latents = np.hstack((capacity, demand, cost))
        else:
            latents = np.hstack((capacity, demand))
        if self.norm:
            latents = (latents - latents.mean(axis=0))/latents.std(axis=0)
        self.latents = torch.Tensor(latents).float()

        allocations = self.data.allocation.values
        if obs_slack:
            unused_capacity = self.data.unused_capacity.values
            unserved_demand = self.data.unserved_demand.values
            obs = np.hstack((allocations.reshape(allocations.shape[0], -1), unused_capacity, unserved_demand))
        else:
            obs = allocations
        if self.norm:
            obs = (obs - obs.mean(axis=0))/obs.std(axis=0)
        self.obs = torch.Tensor(obs).float()

        capacity_perturbation = self.data.capacity_perturbation.values
        demand_perturbation = self.data.demand_perturbation.values
        action = np.hstack((capacity_perturbation, demand_perturbation))
        self.action = torch.Tensor(action).float()

    def __len__(self):
        return len(self.data.coords['sample'])

    def __getitem__(self, index):
        obs = self.obs[index]
        action = self.action[index]
        latents = self.latents[index]

        return obs, action, latents
