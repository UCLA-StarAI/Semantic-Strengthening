from abc import ABC, abstractmethod
import random
import time
import copy
from scipy.special import comb as num_comb

import torch
import torch.nn.functional as F

from logger import Logger
from models_wc import get_model
from utils import AverageMeter, optimizer_from_string, customdefaultdict
from . import metrics
from .metrics import compute_metrics
import numpy as np
from collections import defaultdict
from pathlib import Path

def get_trainer(trainer_name):
    trainers = {"Baseline": BaselineTrainer}
    return trainers[trainer_name]

import pysdd
import itertools

# Semantic Loss
from compute_mpe import CircuitMPE
import sys

timings = None

sys.path.insert(0, '../pypsdd')
inner = CircuitMPE('data/inner.vtree', 'data/inner.sdd')
start_or_end = CircuitMPE('data/start_or_end.vtree', 'data/start_or_end.sdd')
delta = [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]

cuda1 = torch.device('cuda:0')
DIM=12

import sys
from circuit_utils.logexp import *
from circuit_utils.circuit import *
from recover import *

from functools import partial
print = partial(print, flush=True)

# Assume the nodes of the grids are independent
# The constraint is as follows:
#1. The start and end nodes must have exactly 1 neighbor
#2. Every other node, if on, must have exactly 2 neighbors
def compute_indepedent_constraints(outputu):
    dim = int(len(outputu)**0.5)

    sl = torch.tensor(0.0).cuda()
    for i in range(dim):
        for j in range(dim):

            # Literal weights used for semantic loss
            lit_weights = []
            lit_weights.append([1.0 - outputu[i*dim+j], outputu[i*dim+j]])

            # Enumerate all *possible* - valid and invalid - neighbors
            for dx, dy in delta:
                ii = i + dx
                jj = j + dy

                # If the new coordinate is invalid, we set the corresponding
                # variables to false
                if not (ii >= 0 and jj >= 0 and ii < dim and jj < dim):
                    lit_weights.append([torch.tensor(1.0, device=cuda1).expand(outputu[0].shape), torch.tensor(0., device=cuda1).expand(outputu[0].shape)])
                else:
                    lit_weights.append([1.0 - outputu[ii*dim+jj], outputu[ii*dim+jj]])

            # Pick the constraint to use
            if (i == 0 and j == 0) or (i == dim-1 and j == dim-1):
                sl -= start_or_end.get_tf_ac(lit_weights).log().mean()

            else:
                sl -= inner.get_tf_ac(lit_weights).log().mean()
    return sl.sum()


def sdd_idx(i, j, dim=DIM):
    return dim*i+j+1

def get_neighbors(i, j, dim=DIM):
    nbrs = []
    delta = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for dx, dy in delta:
        ii = i + dx
        jj = j + dy

        if ii >= 0 and jj >= 0 and ii < dim and jj < dim:
            nbrs.append((ii,jj))

    return nbrs

var_map = {}
def create_constraints():
    print("Start creating constraints...")
    global sdd
    sdd = pysdd.sdd.SddManager(var_count=DIM**2, auto_gc_and_minimize=True)
    dim = DIM

    constraints = []
    for i in range(DIM):
        for j in range(DIM):
            
            # Start and end points need to have exactly one neighbor
            if (i == 0 and j == 0) or (i == dim-1 and j == dim-1):
                k = 1
            # Every other vertx needs to have exactly two neighbors
            else:
                k = 2

            # To specify that exactly k out of n variables must be true
            # we need to have two types of clauses:
            #1. All (k + 1)-tuples (¬𝑋1 ∨ ¬𝑋2 ∨ … ∨ ¬𝑋𝑘+1)
            # which prevent more than K variables to be true
            #2. All (n-k+1)-tuples which force at least K variables to be true (𝑋𝑗1∨𝑋𝑗2∨…∨𝑋𝑗𝑛−𝑘+1)

            alpha = sdd.true()
            alpha.ref()

            nbrs = [sdd.vars[sdd_idx(i, j)] for i,j in get_neighbors(i, j)]

            # At most k out of n variables is true
            for comb in itertools.combinations(nbrs, k+1):
                clause = -sdd.vars[sdd_idx(i,j)]
                clause.ref()
                for elem in comb:
                    old_clause = clause
                    clause = clause | -elem
                    clause.ref()
                    old_clause.deref()

                old_alpha = alpha
                alpha = alpha & clause
                alpha.ref()
                old_alpha.deref()

            # At least k out of the n variables is true
            for comb in itertools.combinations(nbrs, len(nbrs)-k+1):
                clause = -sdd.vars[sdd_idx(i,j)]
                clause.ref()
                for elem in comb:
                    old_clause = clause
                    clause = clause | elem
                    clause.ref()
                    old_clause.deref()

                old_alpha = alpha
                alpha = alpha & clause
                alpha.ref()
                old_alpha.deref()

            constraints.append(alpha)
            var_map[alpha] = set(nbrs + [sdd.vars[sdd_idx(i,j)]])

    assert(len(var_map) == DIM**2)
    assert(len(constraints) == DIM**2)
    print("End creating constraints.")
    return constraints

# Create constraints
constraints = create_constraints()

class ShortestPathAbstractTrainer(ABC):
    def __init__(
        self,
        *,
        train_iterator,
        test_iterator,
        metadata,
        use_cuda,
        batch_size,
        optimizer_name,
        optimizer_params,
        model_params,
        fast_mode,
        neighbourhood_fn,
        preload_batch,
        lr_milestone_1,
        lr_milestone_2,
        use_lr_scheduling,
        log_path,
        sl_weight,
        K=20,
        interval=5
    ):

        self.fast_mode = fast_mode
        self.use_cuda = use_cuda
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.test_iterator = test_iterator
        self.train_iterator = train_iterator
        self.metadata = metadata
        self.grid_dim = int(np.sqrt(self.metadata["output_features"]))
        self.neighbourhood_fn = neighbourhood_fn
        self.preload_batch = preload_batch
        self.log_path = log_path
        self.sl_weight=sl_weight
        self.interval = interval
        self.K=K
        self.last_batch_time = 0
        self.max_patience = 100
        self.patience = self.max_patience


        self.model = None
        self.build_model(**model_params)

        if self.use_cuda:
            self.model.to("cuda")
        self.optimizer = optimizer_from_string(optimizer_name)(self.model.parameters(), **optimizer_params)
        self.use_lr_scheduling = use_lr_scheduling
        if use_lr_scheduling:
            self.scheduler = MultiStepLR(self.optimizer, milestones=[lr_milestone_1, lr_milestone_2], gamma=0.1)
        self.epochs = 0
        self.recover = 0
        self.train_logger = Logger(scope="training", default_output="tensorboard")
        self.val_logger = Logger(scope="validation", default_output="tensorboard")
        self.best_performance = (0,0)

    def train_epoch(self):
        self.epochs += 1
        batch_time = AverageMeter("Batch time")
        data_time = AverageMeter("Data time")
        avg_loss = AverageMeter("Loss")
        avg_accuracy = AverageMeter("Accuracy")
        avg_perfect_accuracy = AverageMeter("Perfect Accuracy")

        avg_metrics = customdefaultdict(lambda k: AverageMeter("train_"+k))

        if  self.sl_weight and self.recover % self.interval == 0 and self.epochs != 1 and self.last_batch_time < 20: #5:#self.last_batch_time < 1.2 and self.epochs < 42:
            self.model.eval()
            recover_iterator = self.train_iterator.get_epoch_iterator(batch_size=1000, number_of_epochs=1, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)
            batch = next(recover_iterator)

            input = batch["images"]
            label = batch["labels"]

            with torch.no_grad():
                print("Start recovering constraints....")

                self.optimizer.zero_grad(set_to_none=True)
                global constraints; global timings
                if self.epochs > 130:
                    constraints = recover(constraints, input, self.model, K=self.K, var_map=var_map, timings=timings, caution=False, threshold=0.1)
                else:
                    constraints = recover(constraints, input, self.model, K=self.K, var_map=var_map, timings=timings, caution=True)
                print("End recovery... " + str(len(constraints)) + " remaining.")
                #print constraint vars:
                for constraint in constraints:
                    print(var_map[constraint])

                print("Minimizing SDD size...")
                sdd.garbage_collect()
                sdd.minimize_limited()
                print("Done with minimization.")

                flat_target = label.view(label.size()[0], -1)
                lit_weights = [[1-p, p] for p in flat_target.unbind(dim=-1)]

                wmcs = []
                timings = dict()
                for c in constraints:
                    start = time.time()
                    wmcs.append(-wmc(c, lit_weights, log_space=False).log())
                    timings[c] = time.time() - start
                semantic_loss = torch.stack(wmcs)
                assert((semantic_loss == 0).all())

                print(timings)

                # Re-initialize optimizer
                #self.optimizer = optimizer_from_string(self.optimizer_name)(self.model.parameters(), **self.optimizer_params)

        self.recover+=1

        # Start training in earnest
        self.model.train()
        iterator = self.train_iterator.get_epoch_iterator(batch_size=self.batch_size,
                number_of_epochs=1, device='cuda' if self.use_cuda else 'cpu',
                preload=self.preload_batch)

        # Keep track of epoch time
        end = time.time()

        for i, data in enumerate(iterator):
            input, true_path, true_weights = data["images"], data["labels"],  data["true_weights"]

            # measure data loading time
            data_time.update(time.time() - end)

            loss, accuracy, last_suggestion = self.forward_pass(input, true_path, train=True, i=i)

            avg_loss.update(loss.item(), input.size(0))
            avg_accuracy.update(accuracy.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        self.last_batch_time = batch_time.avg
        meters = [batch_time, data_time, avg_loss, avg_accuracy]
        meter_str = "\t".join([str(meter) for meter in meters])
        print(f"Epoch: {self.epochs}\t{meter_str}")

        if self.use_lr_scheduling:
            self.scheduler.step()

        self.train_logger.log(avg_loss.avg, "loss")
        self.train_logger.log(avg_accuracy.avg, "accuracy")

        return {
            "train_loss": avg_loss.avg,
            "train_accuracy": avg_accuracy.avg,
            **{"train_"+k: avg_metrics[k].avg for k in avg_metrics.keys()}
        }

    @torch.no_grad()
    def evaluate(self, print_paths=False):
        avg_metrics = defaultdict(AverageMeter)

        self.model.eval()

        iterator = self.test_iterator.get_epoch_iterator(batch_size=self.batch_size, 
                number_of_epochs=1, shuffle=False, device='cuda' if self.use_cuda else 
                'cpu', preload=self.preload_batch)

        for i, data in enumerate(iterator):
            input, true_path, true_weights = (
                data["images"].contiguous(),
                data["labels"].contiguous(),
                data["true_weights"].contiguous(),
            )

            if self.use_cuda:
                input = input.cuda()
                true_path = true_path.cuda()

            loss, accuracy, last_suggestion = self.forward_pass(input, true_path, train=False, i=i)
            suggested_path = last_suggestion["suggested_path"]
            data.update(last_suggestion)
            if i == 0:
                indices_in_batch = random.sample(range(self.batch_size), 4)
                for num, k in enumerate(indices_in_batch):
                    self.log(data, train=False, k=k, num=num)

            evaluated_metrics = metrics.compute_metrics(true_paths=true_path,
            suggested_paths=suggested_path, true_vertex_costs=true_weights)
            avg_metrics["loss"].update(loss.item(), input.size(0))
            avg_metrics["accuracy"].update(accuracy.item(), input.size(0))

            for key, value in evaluated_metrics.items():
                avg_metrics[key].update(value, input.size(0))

        for key, avg_metric in avg_metrics.items():
            self.val_logger.log(avg_metric.avg, key=key)
        avg_metrics_values = dict([(key, avg_metric.avg) for key, avg_metric in avg_metrics.items()])

        current_performance = (avg_metrics_values['below_0.0001_percent_acc'],avg_metrics_values['valid_acc'])
        if current_performance > self.best_performance:
            self.patience = self.max_patience
            self.best_performance = current_performance
            print("Saving best model")
            torch.save({
                        'epoch': self.epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, Path(self.log_path).with_suffix('.pt'))
        else:
            self.patience -= 1
            print("decreasing patience...")
            if self.patience <= 0:
                print("Ran out of patience")
                exit()

        return avg_metrics_values

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @abstractmethod
    def forward_pass(self, input, true_shortest_paths, train, i):
        pass

    def log(self, data, train, k=None, num=None):
        pass

class BaselineTrainer(ShortestPathAbstractTrainer):
    def build_model(self, model_name, arch_params):
        grid_dim = int(np.sqrt(self.metadata["output_features"]))
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, label, train, i):
        
        # Get model logits
        logits = self.model(input)
        output = logits

        # For use in probability computation
        lgoutput = logsigmoid(output)#.clamp(max=-torch.finfo().eps)
        lgoutputu = lgoutput.unbind(dim=-1)

        # For use in CE computation
        output = torch.sigmoid(output)
        outputu = torch.unbind(output, axis=1)

        # Cross-Entropy
        flat_target = label.view(label.size()[0], -1)
        criterion = torch.nn.BCEWithLogitsLoss()#torch.nn.BCELoss()
        loss = criterion(logits, flat_target).clamp(min=0).mean()
        #if loss.isnan().any():
        #    import pdb; pdb.set_trace()
   

        if train:
            # Semantic Loss 
            #semantic_loss = compute_indepedent_constraints(outputu)
            lit_weights = [[1-p, p] for p in outputu]
            semantic_loss = torch.stack([-wmc(c, lit_weights, log_space=False).log() for c in constraints]).sum(dim=0).mean()
            loss += self.sl_weight * semantic_loss

            # Clear gradients, backprop, and take a step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        accuracy = (output.round() * flat_target).sum() / flat_target.sum()

        suggested_path = output.view(label.shape).round()
        last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

        return loss, accuracy, last_suggestion
