import gymnasium as gym
import itertools
import matplotlib.pyplot as plt
import mip
import networkx as nx
import numpy as np

from gymnasium import spaces
import torch

from .util import generate_random_network

class AllocationEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 0.1}

    def __init__(self, n_nodes: int, n_producers: int, n_consumers: int):
        assert n_producers <= n_nodes
        assert n_consumers <= n_nodes

        network_graph = generate_random_network(n_nodes)
        self.producers = self.np_random.choice(network_graph.nodes, size=n_producers, replace=False)
        self.consumers = self.np_random.choice(network_graph.nodes, size=n_consumers, replace=False)

        self.demand = torch.distributions.Gamma(
            5.*torch.ones_like(self.consumers),
            0.1*torch.ones_like(self.consumers),
        )

        cost_loc = []
        for i, j in itertools.product(self.producers, self.consumers):
            cost_loc.append(nx.dijkstra_path_length(network_graph, i, j, weight='cost'))


        self.assignment_graph = nx.Graph()
        for n in self.producers:
            self.assignment_graph.add_node(n, capacity_loc=5., capacity_scale=1.)

        for n in self.consumers:
            self.assignment_graph.add_node(n, demand_loc=5., demand_scale=1.)

        for i, j in itertools.product(self.producers, self.consumers):
            cost_loc = nx.dijkstra_path_length(network_graph, i, j, weight='cost')
            self.assignment_graph.add_edge(i, j, cost_loc=cost_loc, cost_scale=1.)

        # the actions set the capacity levels for each producer
        self.action_space = spaces.Box(0, np.inf, shape=(self.n_producers, ), dtype=float)

        # there is a capacity constraint for each producer and a demand constraint for each consumer
        n_constrs = self.n_producers + self.n_consumers

        # there are allocations from every producer to every consumer
        n_decisions = self.n_producers * self.n_consumers

        self.observation_space = spaces.Dict({
            'y': spaces.Box(0, np.inf, shape=(self.n_decisions, ), dtype=float)
        })
        self.latent_space = spaces.Dict({
            'W': spaces.Box(0, 1, shape=(n_constrs, n_decisions), dtype=int),
            'h': spaces.Box(0, np.inf, shape=(n_constrs, ), dtype=float),
            'q': spaces.Box(0, np.inf, shape=(n_decisions, ), dtype=float),
        })

    @property
    def n_producers(self):
        return len(self.producers)

    @property
    def n_consumers(self):
        return len(self.consumers)

    def sample_params(self):
        capacity = []
        for n in self.producers:
            if self.stochastic_capacity:
                capacity = self.np_random.gamma(self.)
        capacity_loc = nx.get_node_attributes(self.assignment_graph, "capacity_loc")
        if self.stochastic_capacity:

            capacity = [self.self.np_random.gamma(capacity_loc, capacity_scale)]
            capacity_scale = nx.get_node_attributes(self.assignment_graph, "capacity_scale")
            capacity = self.np_random.gamma(capacity_loc, capacity_scale)
        else:
            capacity = capacity_loc

        demand_loc = nx.get_node_attributes(self.assignment_graph, "demand_loc")
        if self.stochastic_demand:
            demand_scale = nx.get_node_attributes(self.assignment_graph, "demand_scale")
            demand = self.np_random.gamma(demand_loc, demand_scale)
        else:
            demand = demand_loc

        cost_loc = nx.get_edge_attributes(self.assignment_graph, "cost_loc")
        if self.stochastic_cost:
            cost_scale = nx.get_node_attributes(self.assignment_graph, "cost_scale")
            cost = self.np_random.gamma(cost_loc, cost_scale)
        else:
            cost = cost_loc

        return capacity, demand, cost

    def step(self, action):
        capacity = {n: v for n, v in zip(self.producers, action)}
        nx.set_node_attributes(self.assignment_graph, capacity, name="capacity_loc")

        capacity, demand, cost = self.sample_params()
        allocation = self.solve_allocation_problem(capacity, demand, cost)

    def solve_allocation_problem(self, capacity, demand, cost):
        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.verbose = 0

        # add an extra decision for unserved demand
        y = model.add_var_tensor((self.n_producers * self.n_consumers + 1), name='y', var_type='C', lb=0)
        for i, j in itertools.product(self.producers, self.consumers):
            pass

        return model
