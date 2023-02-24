import torch

def as_list(alpha, first_call=True):
    if alpha.bit(): return
    alpha.set_bit(1)

    if alpha.is_decision():
        for p, s in alpha.elements():
            for node in as_list(p, first_call=False): yield node
            for node in as_list(s, first_call=False): yield node
    yield alpha

    if first_call:
        clear_bits_pysdd(alpha)

def clear_bits_pysdd(alpha):
    if alpha.bit() == 0: return
    alpha.set_bit(0)

    if alpha.is_decision():
        for p, s in alpha.elements():
            clear_bits_pysdd(p)
            clear_bits_pysdd(s)

def wmc(alpha, lit_weights, log_space=True):
    d = {}
    for node in as_list(alpha):

        # Use Cache
        data = d.get(node.id)
        if data is not None:
            continue

        if node.is_false():
            data = torch.tensor(0.0 if not log_space else -float('inf'), device=torch.cuda.current_device())

        elif node.is_true():
            data = torch.tensor(1.0 if not log_space else 0.0, device=torch.cuda.current_device())

        elif node.is_literal():
            data = lit_weights[abs(node.literal) - 1][int(node.literal>0)]

        else:
            if log_space:
                data = torch.stack([d[p.id] + d[s.id] for p, s in node.elements()], dim=-1).logsumexp(dim=-1).clamp(max=-1.1920928955078125e-07)
            else:
                data = sum([d[p.id]*d[s.id] for p, s in node.elements()])
        d[node.id] = data
    return data
