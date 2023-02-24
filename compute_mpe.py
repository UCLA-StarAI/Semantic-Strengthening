import sys
sys.path.append("pypsdd")

from pypsdd import Vtree, SddManager, PSddManager, SddNode, Inst, io
from pypsdd import UniformSmoothing, Prior
from pysdd import sdd

""" This is the main way in which SDDs should be used to compute semantic loss.
Construct an instance from a given SDD and vtree file, and then use the available
functions for computing the most probable explanation, weighted model count, or
constructing a tensorflow circuit for integrating semantic loss into a project.
"""


class CircuitMPE:
    def __init__(self, vtree_filename, sdd_filename):

        # Load the Sdd using pysdd
        vtree = sdd.Vtree.from_file(vtree_filename)
        manager = sdd.SddManager.from_vtree(vtree)
        self.alpha = manager.read_sdd_file(sdd_filename.encode())

        # Load the PSdd using pypsdd
        vtree = Vtree.read(vtree_filename)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd_filename, manager)

        # Convert to psdd
        pmanager = PSddManager(vtree)

        # Storing psdd
        self.beta = pmanager.copy_and_normalize_sdd(alpha, vtree)

    def conjoin(self, other):
        gamma = self.alpha.conjoin(other.alpha)
        vtree = Vtree.read(vtree_file)
        manager = SddManager(vtree)


        # Recreate PSDD
        
    def compute_mpe_inst(self, lit_weights, binary_encoding=True):
        mpe_inst = self.beta.get_weighted_mpe(lit_weights)[1]
        if binary_encoding:
            # Sort by variable, but ignoring negatives
            mpe_inst.sort(key=lambda x: abs(x))
            return [int(x > 0) for x in mpe_inst]
        else:
            return mpe_inst

    def weighted_model_count(self, lit_weights):
        return self.beta.weighted_model_count(lit_weights)

    def get_norm_ac(self, litleaves):
        return self.beta.generate_normalized_ac(litleaves)

    def get_tf_ac(self, litleaves):
        return self.beta.generate_tf_ac(litleaves)

    def get_torch_ac(self, litleaves):
        return self.beta.generate_normalized_torch_ac(litleaves)

    def generate_torch_ac_stable(self, litleaves):
        return self.beta.generate_normalized_torch_ac_stable(litleaves)

    # Mainly used for debugging purposes
    def pr_inst(self, inst):
        return self.beta.pr_inst(inst)
    
    def entropy_kld(self):
        import math

        pmanager = PSddManager(self.vtree)
        gamma = pmanager.copy_and_normalize_sdd(self.alpha, self.vtree)
        prior = UniformSmoothing(1.0)
        prior.initialize_psdd(gamma)

        # log model_count(beta) - ent(beta)
        kld = self.beta.kl_psdd(gamma)
        mc = self.beta.model_count()
        entropy = -kld + math.log(mc)

        return entropy

    def Shannon_entropy(self):
        return self.beta.Shannon_entropy()

    def Shannon_entropy_stable(self):
        return self.beta.Shannon_entropy_stable()

    def get_models(self):
        return self.beta.models(self.vtree)

def iter(sdd):
    if sdd.is_decision():
        for p, s in sdd.elements():
            for node in iter(p): yield node
            for node in iter(s): yield node
    yield sdd

if __name__ == '__main__':
    import torch
    from torch import log
    torch.set_printoptions(precision=8)

    # Start test interface between pysdd and pypsdd
    # Read in pysdd vtree and sdd
    from pysdd import sdd
    pysdd_vtree = sdd.Vtree.from_file('abcd_constraint.vtree')
    pysdd_manager = sdd.SddManager.from_vtree(pysdd_vtree)
    pysdd_alpha = pysdd_manager.read_sdd_file('abcd_constraint.sdd'.encode())

    # Read in pypsdd vtree and ensure it matches one converted from pysdd
    vtree_file = Vtree.read('abcd_constraint.vtree')
    vtree = io.vtree_from_pysdd(pysdd_vtree)
    assert(vtree == vtree_file)

    # Create manager
    manager = SddManager(vtree)


    # Read in pypsdd sdd and ensure it matches one converted from pysdd
    alpha_file = io.sdd_read('abcd_constraint.sdd', manager)
    alpha = io.sdd_from_pysdd(pysdd_alpha, manager)
    assert(alpha.is_eq(alpha_file))

    ## End test interface between pysdd and pypsdd
    #exit()

    import pdb; pdb.set_trace()

    # Convert to psdd
    pmanager = PSddManager(vtree)

    # Storing psdd
    beta = pmanager.copy_and_normalize_sdd(alpha, vtree)
    prior = UniformSmoothing(1.0)
    prior.initialize_psdd(beta)

    # An sdd for the formula (a & b) | (c & d)
    c = CircuitMPE('abcd_constraint.vtree', 'abcd_constraint.sdd')

    # literal weights are of the form [[-a, a], [-b, b], [-c, c], [-d, d]]
    lit_weights = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=torch.cuda.current_device())

    # Weighted model counts of both the normalized and unnormalized circuits
    wmc = c.get_norm_ac(lit_weights)

    print(c.entropy_kld())

    # Test 1
    # An sdd for the formula (a & b) | (c & d)
    c = CircuitMPE('abcd_constraint.vtree', 'abcd_constraint.sdd')

    # literal weights are of the form [[-a, a], [-b, b], [-c, c], [-d, d]]
    lit_weights = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=torch.cuda.current_device())

    # Weighted model counts of both the normalized and unnormalized circuits
    wmc = c.get_tf_ac(lit_weights)
    wmc_normalized = c.get_torch_ac(lit_weights)

    # assert the wmc of the normalized and unnormalized circuits match
    assert(c.get_tf_ac(lit_weights) == 0.0976)
    assert(c.get_torch_ac(lit_weights) == 0.0976)
    
    # Entropy of the probability distribution
    weights = torch.tensor([0.0224, 0.0096, 0.0056, 0.0324, 0.0036, 0.0216, 0.0024], device=torch.cuda.current_device())
    probs = weights/wmc
    entropy = -sum([p*log(p) for p in probs])

    # Circuit Entropy
    circuit_entropy = c.Shannon_entropy()
    print(circuit_entropy)
    print(entropy)
    exit()

    # Assert the circuit's entropy and the entropy of the groundtruth distribution match
    assert(torch.isclose(circuit_entropy, entropy))

    # Check probabilities of all the models of the formula
    assert(torch.isclose(c.pr_inst([-1, -2, 3, 4]), torch.tensor(0.2295), atol=1e-04))
    assert(torch.isclose(c.pr_inst([-1, 2, 3, 4]), torch.tensor(0.0984), atol=1e-04))
    assert(torch.isclose(c.pr_inst([1, -2, 3, 4]), torch.tensor(0.0574), atol=1e-04))
    assert(torch.isclose(c.pr_inst([1, 2, -3, -4]), torch.tensor(0.3320), atol=1e-04))
    assert(torch.isclose(c.pr_inst([1, 2, -3, 4]), torch.tensor(0.0369), atol=1e-04))
    assert(torch.isclose(c.pr_inst([1, 2, 3, -4]), torch.tensor(0.2213), atol=1e-04))
    assert(torch.isclose(c.pr_inst([1, 2, 3, 4]), torch.tensor(0.0246), atol=1e-04))


    # Test 2
    # An sdd for the formula true
    c = CircuitMPE('abcd_constraint.vtree', 'true_constraint.sdd')

    # literal weights are of the form [[-a, a], [-b, b], [-c, c], [-d, d]]
    lit_weights = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=torch.cuda.current_device())

    models = [[0, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 1, 1],
             [0, 1, 0, 0],
             [0, 1, 0, 1],
             [0, 1, 1, 0],
             [0, 1, 1, 1],
             [1, 0, 0, 0],
             [1, 0, 0, 1],
             [1, 0, 1, 0],
             [1, 0, 1, 1],
             [1, 1, 0, 0],
             [1, 1, 0, 1],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]

    probs = []
    for model in models:
        prob = 1
        for i, val in enumerate(model):
            prob *= lit_weights[i][val]
        probs += [prob]

    # Weighted model counts of both the normalized and unnormalized circuits
    wmc = c.get_tf_ac(lit_weights)
    wmc_normalized = c.get_torch_ac(lit_weights)

    assert(wmc == wmc_normalized == 1)

    # Brute force entropy
    entropy = -sum([p*log(p) for p in probs])

    # Circuit Entropy
    circuit_entropy = c.Shannon_entropy()
    assert(circuit_entropy == entropy)

    # Test 3
    # An sdd for the formula (P | L) & (-A | P) & (-K | (A | L))
    c = CircuitMPE('LKPA_constraint.vtree', 'LKPA_constraint.sdd')

    # literal weights form     [[-L,    L], [-K,    K], [-P,    P], [-A,    A]]
    lit_weights = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=torch.cuda.current_device())

    # Weighted model counts of both the normalized and unnormalized circuits
    wmc = c.get_tf_ac(lit_weights)
    wmc_normalized = c.get_torch_ac(lit_weights)

    # assert the wmc of the normalized and unnormalized circuits match
    print(c.get_tf_ac(lit_weights))
    print(c.get_torch_ac(lit_weights))
    assert(c.get_tf_ac(lit_weights) == 0.4216)
    assert(c.get_torch_ac(lit_weights) == 0.4216)
    
    # Entropy of the probability distribution
    weights = torch.tensor([0.2016, 0.0224, 0.0096, 0.0756, 0.0504, 0.0056, 0.0324, 0.0216, 0.0024], device=torch.cuda.current_device())
    probs = weights/wmc
    entropy = -sum([p*log(p) for p in probs])

    # Circuit Entropy
    circuit_entropy = c.Shannon_entropy()

    # Assert the circuit's entropy and the entropy of the groundtruth distribution match
    assert(torch.isclose(circuit_entropy, entropy))
