import torch
import itertools
from circuit_utils.logexp import *
from circuit_utils.circuit import * 

def mi(c1, c2, lit_weights):
    """
                 c2            -c2
          +-----------+------------+
          |           |            |
     c1   |     a     |      b     |
          |           |            |
          +------------------------+
          |           |            |
    -c1   |     c     |      d     |
          |           |            |
          +-----------+------------+
    """
    #TODO
    #if cache[c1+c2]:
    # Conjoin the two constraints
    c1_c2 = c1 & c2
    c1_c2.ref()

    # marginals
    p_c1 = wmc(c1, lit_weights) #a+b
    p_c2 = wmc(c2, lit_weights) #a+c

    # Calculate the probabilities: a, b, c, and d
    p_c1_c2 = wmc(c1_c2, lit_weights) #a
    p_c1_nc2 = logsubexp(p_c1, p_c1_c2) #b
    p_nc1_c2 = logsubexp(p_c2, p_c1_c2) #c
    tmp = torch.stack((p_c1_c2, p_c1_nc2,  p_nc1_c2), dim=-1).logsumexp(dim=-1).clamp(max=-1.1920928955078125e-07)
    p_nc1_nc2 = log1mexp(-tmp) #d


    p_c1 = [log1mexp(-p_c1), p_c1]
    p_c2 = [log1mexp(-p_c2), p_c2]
    p_c1_c2 = [[p_nc1_nc2, p_nc1_c2],[p_c1_nc2, p_c1_c2]]

    mi = 0.0
    for x, y in itertools.product([0,1], repeat=2):
        a = p_c1_c2[x][y].exp()
        b = p_c1_c2[x][y] - (p_c1[x] + p_c2[y])
        mi += xty(a, b)

        if torch.any(torch.isnan(mi)) or torch.any(torch.isinf(mi)):
            import pdb; pdb.set_trace()
            print("Crap! nan or inf")

    mi.clamp(min=0)
    return (mi.mean().item(), set([c1_c2]))

def recover(constraints, input, model, is_input, K=5, var_map=None, limit=100000):

    # Estimate probability of the constraints using
    # training data
    #output = model(input)
    #outputu = logsigmoid(output).unbind(axis=1)
    #outputu = [[log1mexp(-p), p] for p in outputu]

    output = model(input.view(-1, 9, 9, 9))
    output = F.softmax(output.view(-1, 9), dim=-1).view(-1, 9*9*9)
    output = torch.where(is_input.bool().view(1000, -1), torch.tensor(0.).cuda(), output)
    output = torch.where((input == 1).view(-1, 9*9*9),  torch.tensor(1.).cuda(), output)
    outputu = torch.unbind(output, axis=1)
    outputu = [[log1mexp(-p), p] for p in outputu]

    # pairwise mutual information
    pwmi = []
    for i, (c1, c2) in enumerate(itertools.combinations(constraints, 2)):

        # If the constraints are defined over
        # disjoint sets of variables, they are
        # independent
        if var_map[c1].isdisjoint(var_map[c2]):
            continue

        pwmi.append([*mi(c1, c2, outputu), set([c1, c2])])

    # Merge the K constraints with highest MI
    pwmi = sorted(pwmi, key=lambda tup: tup[0], reverse=True)
    to_merge = pwmi[:K]

    # Consolidate common constraints
    out = []
    while len(to_merge) > 0:
        curr, *rest = to_merge

        lcurr = -1
        while len(curr[-1]) > lcurr:
            lcurr = len(curr[-1])

            rest_ = []
            for other in rest:
                if not curr[-1].isdisjoint(other[-1]):
                    print("Consolidated")
                    curr[-2] |= other[-2]
                    curr[-1] |= other[-1]
                else:
                    rest_.append(other)
            rest = rest_

        out.append(curr)
        to_merge = rest

    to_merge = out
    print(len(to_merge))

    for i in range(len(to_merge)):
        for j in range(len(to_merge)):
            if i != j:
                assert (to_merge[i][-1].isdisjoint(to_merge[j][-1]))

    # Dereference unused constraints
    for elem in pwmi[K:]:
        for e in elem[1]: e.deref()

    # A set to track the constraints to be removed
    to_remove = set()
    for _, joint, indiv in to_merge:

        # if we have decided to model c1 and c2 jointly,
        # and c1 and c3 jointly, then we conjoin the two
        # constraints to get a single constraint for
        # c1 & c2 & c3
        first, *rest = joint
        for r in rest:
            old_first = first
            first = first & r
            first.ref()
            old_first.deref()

        # append the resulting constraint the list of independent
        # constraints
        constraints.append(first)

        # Update the constraints to remove
        to_remove |= indiv

        # Update the variables the new constraint is defined over
        variables = set()
        for c in indiv:
            variables |= var_map[c]
        var_map[first] = variables

    # Remove and dereference constraints subsumed by newer ones
    for c in to_remove:
        var_map[c].pop()
        constraints.remove(c)
        c.deref()

    return constraints
