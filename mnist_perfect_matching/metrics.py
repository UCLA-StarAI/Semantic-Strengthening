import numpy as np
from decorators import input_to_numpy, none_if_missing_arg
from utils import all_accuracies
from comb_modules.utils import edges_from_grid
from collections import Counter
from functools import partial
import itertools

@none_if_missing_arg
def perfect_match_accuracy(true_matching, suggested_matching):
    matching_correct = np.sum(np.abs(true_matching - suggested_matching), axis=-1)
    avg_matching_correct = (matching_correct < 0.5).mean()
    return avg_matching_correct


@none_if_missing_arg
def cost_ratio(edge_costs, true_matching, suggested_matching):
    suggested_matching_edge_costs = suggested_matching * edge_costs
    true_matching_edge_costs = true_matching * edge_costs
    #print(np.sum(suggested_matching_edge_costs, axis=1) / np.sum(true_matching_edge_costs, axis=1))
    return (np.sum(suggested_matching_edge_costs, axis=1) / np.sum(true_matching_edge_costs, axis=1)).mean()


@input_to_numpy
def compute_metrics(true_edge_costs, true_matching, suggested_matching, true_vertex_costs, always_valid):
    is_valid_label = partial(is_valid_matching, grid_dim=len(true_vertex_costs[0]), always_valid=always_valid)
    #print(is_valid_label)
    metrics = {
        "perfect_match_accuracy": perfect_match_accuracy(true_matching, suggested_matching),
        "cost_ratio_suggested_true": cost_ratio(true_edge_costs, true_matching, suggested_matching),
        **all_accuracies(true_matching,suggested_matching, true_edge_costs, is_valid_label,6)
    }
    return metrics

def is_valid_matching(matching, grid_dim, always_valid):
    if always_valid: return True
    cnt = Counter()
    edges = edges_from_grid(grid_dim, '4-grid')
    for i, (x1,y1,x2,y2) in itertools.compress(enumerate(edges),matching):
        cnt[(x1,y1)]+=1
        cnt[(x2,y2)]+=1
        if cnt[(x1,y1)] > 1 or cnt[(x2,y2)] > 1:
            return False
    return np.sum(matching.astype(np.int32)) == int(grid_dim**2/2)
