import gymnasium as gym
import itertools
import mip
import networkx as nx
import numpy as np
import torch

from gymnasium import spaces

from .util import generate_random_network

class AllocationDataset(torch.utils.data.Dataset):
    def __init__(self, n_nodes: int, n_producers: int, n_consumers: int, n_samples: int):
        super().__init__()

        self.length = n_samples

        self.env = AllocationEnv(n_nodes, n_producers, n_consumers)
        self.x = torch.distributions.Uniform(1, 10).sample((n_samples, n_producers))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        action = self.x[index]
        observation, reward, _, _, latents = self.env.step(action)
        return action, observation['y'], latents['W'], latents['h'], latents['q'], reward


class AllocationEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 0.1}

    def __init__(self, n_nodes: int, n_producers: int, n_consumers: int):
        super().__init__()

        assert n_producers <= n_nodes
        assert n_consumers <= n_nodes

        network_graph = generate_random_network(n_nodes)
        self.producers = self.np_random.choice(network_graph.nodes, size=n_producers, replace=False)
        self.consumers = self.np_random.choice(network_graph.nodes, size=n_consumers, replace=False)

        self.demand_dist = torch.distributions.Gamma(
            5.*torch.ones(self.consumers.shape),
            1*torch.ones(self.consumers.shape),
        )

        cost_loc_list = []
        for i, j in itertools.product(self.producers, self.consumers):
            c = nx.dijkstra_path_length(network_graph, i, j, weight='cost')
            cost_loc_list.append(c)
        cost_loc = 10*torch.tensor(cost_loc_list)
        cost_loc = cost_loc.clamp(1e-8) # gamma loc must be > 0
        self.cost_dist = torch.distributions.Gamma(cost_loc, 1*torch.ones_like(cost_loc))

        # there is a capacity constraint for each producer and a demand constraint for each consumer
        self.n_constrs = self.n_producers + self.n_consumers

        # there are allocations from every producer to every consumer
        self.n_decisions = self.n_producers * self.n_consumers

        self.action_space = spaces.Dict({
            'x': spaces.Box(0, np.inf, shape=(self.n_producers, ), dtype=float)
        })
        self.observation_space = spaces.Dict({
            'y': spaces.Box(0, np.inf, shape=(self.n_decisions, ), dtype=float)
        })
        self.latent_space = spaces.Dict({
            'W': spaces.Box(0, 1, shape=(self.n_constrs, self.n_decisions), dtype=int),
            'h': spaces.Box(0, np.inf, shape=(self.n_constrs, ), dtype=float),
            'q': spaces.Box(0, np.inf, shape=(self.n_decisions, ), dtype=float),
        })

    @property
    def n_producers(self):
        return len(self.producers)

    @property
    def n_consumers(self):
        return len(self.consumers)

    def step(self, capacity):
        capacity = capacity.numpy()
        demand = self.demand_dist.sample().numpy()
        cost = self.cost_dist.sample().clamp(1e-3).numpy() # cost must be > 0 or problem is unbounded

        model = self.formulate_allocation_problem(capacity, demand, cost)
        assert model.optimize() == mip.OptimizationStatus.OPTIMAL, f'Did not find the optimal solution, status was {model.status}'
        y = np.array([var.x for var in model.vars])
        Q = model.objective.x
        W, h, q = self.get_slack_form(model)

        obs = {'y': y}
        latents = {'W': W, 'h': h, 'q': q}

        terminated = True
        truncated = False

        return obs, Q, terminated, truncated, latents

    def formulate_allocation_problem(self, capacity, demand, cost, unused_penalty=0, unsrvd_penalty=1e3):
        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.verbose = 0

        y = model.add_var_tensor((self.n_producers, self.n_consumers), name='y', var_type='C', lb=0)
        y_unused = model.add_var_tensor((self.n_producers, ), name='y_unused', var_type='C', lb=0)
        y_unsrvd = model.add_var_tensor((self.n_consumers, ), name='y_unsrvd', var_type='C', lb=0)

        for n in range(self.n_producers):
            model += mip.xsum(y[n, :]) + y_unused[n] == capacity[n]

        for n in range(self.n_consumers):
            model += mip.xsum(y[:, n]) + y_unsrvd[n] == demand[n]

        model += mip.xsum(cost * y.flatten()) + mip.xsum(unused_penalty*y_unused) + mip.xsum(unsrvd_penalty*y_unsrvd)
        return model

    def get_slack_form(self, model):
        n_constrs = len(model.constrs)
        n_vars = len(model.vars)

        A = np.empty((n_constrs, n_vars))
        b = np.empty(n_constrs)
        c = np.empty(n_vars)

        for i, constr in enumerate(model.constrs):
            assert constr.expr.sense == '=', f'Only = constraints are allowed in slack form, but got a {constr.expr.sense} constraint.'
            b[i] = -constr.expr.const

            for j, var in enumerate(model.vars):
                coeff = constr.expr.expr.get(var, 0)
                A[i, j] = coeff

        for i, var in enumerate(model.vars):
            c[i] = model.objective.expr.get(var, 0)

        return A, b, c
