import torch

def blackjack_cost(c1, c2, limit=11):
    return -(limit - min(c1 + c2, limit))


def torch_blackjack_cost(c1, c2, limit):
    return -(limit - torch.clamp(c1 + c2, min=0, max=limit))


def product_cost(c1, c2):
    return c1 * c2


def torch_product_cost(c1, c2):
    return c1 * c2


def x10_cost(c1, c2):
    return 10 * c1 + c2

def x1_cost(c1, c2):
    return c1 + c2

def torch_x10_cost(c1, c2):
    return 10 * c1 + c2

def torch_x1_cost(c1, c2):
    return c1 + c2

edge_cost_fns = {"1x": x1_cost, "10x": x10_cost, "product": product_cost, "blackjack": blackjack_cost}

torch_edge_cost_fns = {"1x": x1_cost, "10x": torch_x10_cost, "product": torch_product_cost, "blackjack": torch_blackjack_cost}
