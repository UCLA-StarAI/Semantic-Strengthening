import numpy as np
import torch
import itertools
import ray

from utils import maybe_parallelize


def perfect_matching(edges, edge_weights, num_vertices):
    edges = tuple(map(tuple, edges))  # Make hashable
    pm = blossom.PerfectMatching(num_vertices, len(edges))

    for (v1, v2), w in zip(edges, edge_weights):
        pm.AddEdge(int(v1), int(v2), float(w))

    pm.Solve()

    edge_to_index_dict = dict(zip(edges, itertools.count()))
    unique_matched_edges = [(v, pm.GetMatch(v)) for v in range(num_vertices) if v < pm.GetMatch(v)]
    indices = [edge_to_index_dict[edge] for edge in unique_matched_edges]
    solution = np.zeros(len(edges)).astype(np.float32)
    solution[indices] = 1
    return solution


@ray.remote
def parallel_perfect_matching(edges, edge_weights, num_vertices):
    return perfect_matching(edges, edge_weights, num_vertices)


def get_solver(edges, num_vertices):
    def solver(edge_weights):
        return perfect_matching(edges, edge_weights, num_vertices)

    return solver


class PerfectMatchingSolver(torch.autograd.Function):
    def __init__(self, lambda_val, num_vertices, edges):
        self.lambda_val = lambda_val
        self.num_edges = len(edges)
        self.num_vertices = num_vertices
        self.edges = edges
        self.solver = get_solver(edges, num_vertices)

    def forward(self, weights):
        self.weights = weights.detach().cpu().numpy()
        self.perfect_matchings = np.array(maybe_parallelize(self.solver, list(self.weights)))
        return torch.from_numpy(self.perfect_matchings).float().to(weights.device)

    def backward(self, grad_output):
        assert grad_output.shape == self.perfect_matchings.shape
        device = grad_output.device
        grad_output = grad_output.cpu().numpy()

        weights_prime = np.maximum(self.weights + self.lambda_val * grad_output, 0.0)
        better_matchings = np.array(maybe_parallelize(self.solver, list(weights_prime)))

        gradient = -(self.perfect_matchings - better_matchings) / self.lambda_val
        return torch.from_numpy(gradient).to(device), None
