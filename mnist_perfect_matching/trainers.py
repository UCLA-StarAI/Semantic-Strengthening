from comb_modules.utils import cached_vertex_grid_to_edges, cached_vertex_grid_to_edges_grid_coords
import time
from abc import ABC, abstractmethod

import torch
from comb_modules.perfect_matching import PerfectMatchingSolver
from comb_modules.losses import HammingLoss
from comb_modules.utils import edges_from_grid, neighbours_4, vertex_index
from mnist_perfect_matching.edge_costs import torch_edge_cost_fns
from logger import Logger
from models import get_model
from utils import AverageMeter, optimizer_from_string, customdefaultdict
from functools import partial
from decorators import to_tensor, to_numpy
from . import metrics
from .metrics import compute_metrics
import numpy as np
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from pathlib import Path

# Circuit includes
DIM=10
import pysdd
import itertools
from pysdd.sdd import SddManager, Vtree

import sys
from circuit_utils.logexp import *
from circuit_utils.circuit import *
from recover import *

from functools import partial
print = partial(print, flush=True)

# Constraint functions
def edges_from_vertex(x, y, N, neighbourhood_fn):
    v = (x, y)
    neighbours = neighbours_4(*v, x_max=N, y_max=N)
    v_edges = []
    for vn in neighbours:
        if vertex_index(v, N) < vertex_index(vn, N):
            v_edges += [(*v, *vn)]
        else:
            v_edges += [(*vn, *v)]
    return v_edges

#var_map = {}
#def create_constraints():
#    print("Start creating constraints...")
#    global sdd
#    sdd = pysdd.sdd.SddManager(var_count=len(edges_from_grid(DIM, neighbourhood_fn='4-grid')), auto_gc_and_minimize=True)
#
#    edges_to_indices = {tuple(e): i+1 for i, e in enumerate(edges_from_grid(DIM, neighbourhood_fn='4-grid'))}
#
#    constraints = []
#    for x in range(DIM):
#        for y in range(DIM):
#            print(x, y)
#
#            alpha = sdd.true()
#            edges = [sdd.vars[edges_to_indices[e]] for e in edges_from_vertex(x, y, DIM, neighbourhood_fn='4-grid')]
#            
#            # At least one edge
#            clause = sdd.false() if len(edges) > 0 else sdd.true()
#            for edge in edges:
#                old_clause = clause
#                clause = clause | edge
#                clause.ref()
#                old_clause.deref()
#
#            old_alpha = alpha
#            alpha = alpha & clause
#            alpha.ref()
#            old_alpha.deref()
#
#            # At most one edge
#            beta = sdd.true()
#            for comb in itertools.combinations(edges, 2):
#                clause = (-comb[0] | -comb[1])
#                clause.ref()
#
#                beta_old = beta
#                beta = beta & clause
#                beta.ref()
#                beta_old.deref()
#
#            old_alpha = alpha
#            alpha = alpha & beta
#            alpha.ref()
#            old_alpha.deref()
#
#            constraints.append(alpha)
#            var_map[alpha] = set(edges)
#
#    assert(len(var_map) == DIM**2)
#    assert(len(constraints) == DIM**2)
#    print("End creating constraints.")
#    return constraints

var_map = {}
def create_constraints():
    print("Start creating constraints...")
    global sdd
    sdd = pysdd.sdd.SddManager(var_count=len(edges_from_grid(DIM, neighbourhood_fn='4-grid')), auto_gc_and_minimize=True)

    edges_to_indices = {tuple(e): i+1 for i, e in enumerate(edges_from_grid(DIM, neighbourhood_fn='4-grid'))}
    indices_to_edges = {v: k for k, v in edges_to_indices.items()}

    constraints = []
    #gamma = sdd.true()
    for x in range(DIM):
        for y in range(DIM):
            print(x, y)

            #alpha = sdd.true()
            constraint = ""
            edges = [sdd.vars[edges_to_indices[e]] for e in edges_from_vertex(x, y, DIM, neighbourhood_fn='4-grid')]
            real_edges = edges_from_vertex(x, y, DIM, neighbourhood_fn='4-grid')
            
            # At least one edge
            clause = sdd.false() if len(edges) > 0 else sdd.true()
            i = 0
            for edge in edges:
                constraint += str(real_edges[i]) + " | "
                i+=1
                old_clause = clause
                clause = clause | edge
                clause.ref()
                old_clause.deref()

            #old_alpha = alpha
            #alpha = alpha & clause
            #alpha.ref()
            #old_alpha.deref()

            # At most one edge
            beta = sdd.true()
            for comb in itertools.combinations(edges, 2):
                constraint += "& (-" + str(indices_to_edges[comb[0].literal]) + " | " + "-" + str(indices_to_edges[comb[1].literal]) + ")"

                beta_old = beta
                beta = beta & (-comb[0] | -comb[1])
                beta.ref()
                beta_old.deref()


            #print(constraint)
            alpha = clause & beta
            alpha.ref()
            beta.deref()
            clause.deref()
            #old_alpha.deref()
            
            #old_gamma = gamma
            #gamma = gamma & alpha
            #gamma.ref()
            #old_gamma.deref()
            constraints.append(alpha)
            #print(gamma.model_count())
            var_map[alpha] = set(edges)
    print(len(constraints))
    assert(len(var_map) == DIM**2)
    assert(len(constraints) == DIM**2)
    #constraints.append(gamma)
    print("End creating constraints.")
    return constraints

# Create constraints
constraints = create_constraints()
#lit_weights = [[1-p, p] for p in flat_target.unbind(dim=-1)]
#semantic_loss = torch.stack([-wmc(c, lit_weights, log_space=False).log() for c in constraints])
#assert((semantic_loss == 0).all())
#semantic_loss = torch.stack([-wmc(c, flat_target.log().unbind(dim=-1), log_space=True) for c in constraints])

def get_trainer(trainer_name):
    trainers = {"Baseline": BaselineTrainer}
    return trainers[trainer_name]


class PerfectMatchingAbstractTrainer(ABC):
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
        edge_cost_fn_name,
        edge_cost_params,
        fast_mode,
        compute_all_train_metrics,
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
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.test_iterator = test_iterator
        self.train_iterator = train_iterator
        self.metadata = metadata
        self.grid_dim = int(np.sqrt(self.metadata["output_features"]))
        print("The grid dimension:", self.grid_dim)
        self.edge_cost_fn = partial(torch_edge_cost_fns[edge_cost_fn_name], **edge_cost_params)
        self.compute_all_train_metrics = compute_all_train_metrics
        self.always_valid_matching = False
        self.preload_batch = preload_batch
        self.log_path = log_path
        self.sl_weight=sl_weight
        self.interval = interval
        self.K = K
        self.last_batch_time = 0
        self.max_patience = 25
        self.patience = self.max_patience

        self.model = None
        self.build_model(**model_params)

        checkpoint = torch.load('/space/ahmedk/approxback/iclr_logs/pm/10x10/nosl_lr1e-2.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.use_cuda:
            self.model.to("cuda")
        self.optimizer = optimizer_from_string(optimizer_name)(self.model.parameters(), **optimizer_params)
        self.use_lr_scheduling = use_lr_scheduling
        if self.use_lr_scheduling:
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

        if self.compute_all_train_metrics:
            avg_metrics = customdefaultdict(lambda k: AverageMeter("train_"+k))

        # Recover constraints
        if  self.sl_weight and self.recover % self.interval == 0 and self.epochs != 1 and self.last_batch_time < 3:#self.last_batch_time < 1.2 and self.epochs < 42:
            self.model.eval()
            recover_iterator = self.train_iterator.get_epoch_iterator(batch_size=1000, number_of_epochs=1, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)
            batch = next(recover_iterator)

            input = batch["images"]
            label = batch["labels"] 

            with torch.no_grad():
                print("Start recovering constraints....")

                self.optimizer.zero_grad(set_to_none=True)
                global constraints
                constraints = recover(constraints, input, self.model, K=self.K, var_map=var_map)
                print("End recovery... " + str(len(constraints)) + " remaining.")

                print("Minimizing SDD size...")
                sdd.garbage_collect()
                sdd.minimize_limited()
                print("Done with minimization.")

                flat_target = label.view(label.size()[0], -1)
                lit_weights = [[1-p, p] for p in flat_target.unbind(dim=-1)]
                semantic_loss = torch.stack([-wmc(c, lit_weights, log_space=False).log() for c in constraints])
                assert((semantic_loss == 0).all())

                # Re-initialize optimizer
                #self.optimizer = optimizer_from_string(self.optimizer_name)(self.model.parameters(), **self.optimizer_params)

        self.recover+=1

        self.model.train()
        iterator = self.train_iterator.get_epoch_iterator(batch_size=self.batch_size, number_of_epochs=1, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)
        end = time.time()
        for i, data in enumerate(iterator):
            input, true_matching, true_weights = data["images"], data["labels"],  data["true_weights"]
            assert input.device != 'cuda' or self.use_cuda
            #if i == 0:
            #    self.log(data, train=True)

            loss, accuracy, last_suggestion = self.forward_pass(input, true_matching, train=True, i=i)

            #suggested_matching = last_suggestion["suggested_matching"]
            #true_edge_costs = self.vertex_weights_to_edge_weights(true_weights)

            #if self.compute_all_train_metrics:
            #    batch_metrics = metrics.compute_metrics(true_edge_costs=true_edge_costs, true_matching=true_matching,
            #    suggested_matching=suggested_matching, true_vertex_costs=true_weights, always_valid=self.always_valid_matching)
            #    # update batch metrics
            #    {avg_metrics[k].update(v, input.size(0)) for k, v in batch_metrics.items()}
            #    assert len(avg_metrics.keys()) > 0

            avg_loss.update(loss.item(), input.size(0))
            avg_accuracy.update(accuracy.item(), input.size(0))

            # compute gradient and do SGD step
            #self.optimizer.zero_grad()
            #loss.backward()
            #self.optimizer.step()

            # measure elapsed time
            elapsed_time = time.time() - end
            #if(elapsed_time > 14):
            #    pass
            #    #print(elapsed_time)
            batch_time.update(elapsed_time)
            end = time.time()

            if self.fast_mode:
                break

        self.last_batch_time = batch_time.avg
        meters = [batch_time, data_time, avg_loss, avg_accuracy]
        meter_str = "\t".join([str(meter) for meter in meters])
        print(f"Epoch: {self.epochs}\t{meter_str}")

        if self.use_lr_scheduling:
            self.scheduler.step()
        self.train_logger.log(avg_loss.avg, "loss")
        self.train_logger.log(avg_accuracy.avg, "accuracy")

        train_metrics =   {
            "train_loss": avg_loss.avg,
            "train_accuracy": avg_accuracy.avg
        }
        #if self.compute_all_train_metrics:
        #    train_metrics.update({"train_"+k: avg_metrics[k].avg for k in avg_metrics.keys()})

        return train_metrics

    @torch.no_grad()
    def evaluate(self):
        avg_metrics = defaultdict(AverageMeter)

        self.model.eval()

        iterator = self.test_iterator.get_epoch_iterator(batch_size=self.batch_size, number_of_epochs=1, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)
        for i, data in enumerate(iterator):
            input, true_matching, true_weights = (
                data["images"],
                data["labels"],
                data["true_weights"],
            )

            if i == 0:
                self.log(data, train=False)

            loss, accuracy, last_suggestion = self.forward_pass(input, true_matching, train=False, i=i)
            suggested_matching = last_suggestion["suggested_matching"]

            true_edge_costs = self.vertex_weights_to_edge_weights(true_weights).to(true_matching.device)

            evaluated_metrics = compute_metrics(
                true_edge_costs=true_edge_costs, true_matching=true_matching, suggested_matching=suggested_matching,
                true_vertex_costs = true_weights, always_valid=self.always_valid_matching
            )
            avg_metrics["loss"].update(loss.item(), input.size(0))
            avg_metrics["accuracy"].update(accuracy.item(), input.size(0))
            for key, value in evaluated_metrics.items():
                avg_metrics[key].update(value, input.size(0))

            if self.fast_mode:
                break

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

    def vertex_weights_to_edge_weights(self, weights):
        x, y, xn, yn = cached_vertex_grid_to_edges_grid_coords(self.grid_dim)
        edge_costs = self.edge_cost_fn(weights[:, x, y], weights[:, xn, yn])
        return edge_costs

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @abstractmethod
    def forward_pass(self, input, true_matching, train, i):
        pass

    def log(self, data, train):
        pass
        #logger = self.train_logger if train else self.val_logger
        #for k in range(4):
        #    if not train:
        #        img = self.metadata["denormalize"](data["full_images"][k]).squeeze().astype(np.uint8)
        #        if img.shape[0] == 1:
        #            img = np.ones((3, *img.shape)) * 255 * img
        #        logger.log(img, key=f"full_input_{k}", data_type="image")


class BaselineTrainer(PerfectMatchingAbstractTrainer):
    def build_model(self, model_name, arch_params):
        print(arch_params)
        grid_dim = int(np.sqrt(self.metadata["output_features"]))
        self.model = get_model(
            model_name, out_features=len(edges_from_grid(grid_dim, neighbourhood_fn='4-grid')), in_channels=self.metadata["num_channels"], arch_params=arch_params
        )
        self.always_valid_matching = False

    def forward_pass(self, input, label, train, i):

        self.optimizer.zero_grad(set_to_none=True)

        # Get model logits
        output = self.model(input)

        # For use in probability computation
        lgoutput = logsigmoid(output)#.clamp(max=-torch.finfo().eps)
        lgoutputu = lgoutput.unbind(dim=-1)

        # For use in CE computation
        output = torch.sigmoid(output)
        outputu = torch.unbind(output, axis=1)

        # Cross-Entropy
        flat_target = label.view(label.size()[0], -1)
        criterion = torch.nn.BCELoss(reduction='none')
        loss = criterion(output, flat_target).mean()

        if train:
            # to try: mean instead of sum, and re-initializing the optimizer
            if self.sl_weight:
                # Semantic Loss
                #lglit_weights = [[log1mexp(-p.clamp(max=-torch.finfo().eps)), p] for p in lgoutputu]
                #semantic_loss = torch.stack([-wmc(c, lglit_weights, log_space=True) for c in constraints]).sum(dim=0).mean()
                lit_weights = [[1-p, p] for p in outputu]
                semantic_loss = torch.stack([-wmc(c, lit_weights, log_space=False).log() for c in constraints]).sum(dim=0).mean()
                #print("Semantic Loss:", self.sl_weight * semantic_loss)
                loss += self.sl_weight * semantic_loss

            # Clear gradients, backprop, and take a step
            loss.backward()
            self.optimizer.step()

        accuracy = (output.round() * flat_target).sum() / flat_target.sum()

        suggested_matching = output.view(label.shape).round()
        last_suggestion = {"edge_costs": None, "suggested_matching": suggested_matching.cpu().detach().numpy()}

        return loss, accuracy, last_suggestion

        #import pysdd
        #import itertools
        #from pysdd.sdd import SddManager, Vtree
        #grid_dim = int(np.sqrt(self.metadata["output_features"]))
        #sdd = pysdd.sdd.SddManager(var_count=len(edges_from_grid(grid_dim, neighbourhood_fn='4-grid')), auto_gc_and_minimize=True)


        #print(len(edges_from_grid(grid_dim, neighbourhood_fn='4-grid')))
        #edges_to_indices = {tuple(e): i+1 for i, e in enumerate(edges_from_grid(grid_dim, neighbourhood_fn='4-grid'))}
        #alpha = sdd.true()

        #def edges_from_vertex(x, y, N, neighbourhood_fn):
        #    v = (x, y)
        #    neighbours = neighbours_4(*v, x_max=N, y_max=N)
        #    v_edges = []
        #    for vn in neighbours:
        #        if vertex_index(v, N) < vertex_index(vn, N):
        #            v_edges += [(*v, *vn)]
        #        else:
        #            v_edges += [(*vn, *v)]
        #    #v_edges = [
        #    #    (*v, *vn) for vn in neighbours if vertex_index(v, N) < vertex_index(vn, N) else (*vn, *v)
        #    #]  # Enforce ordering on vertices
        #    return v_edges

        #k = 1
        #for x in range(grid_dim):
        #    for y in range(grid_dim):
        #        print(x, y)

        #        edges = [sdd.vars[edges_to_indices[e]] for e in edges_from_vertex(x, y, grid_dim, neighbourhood_fn='4-grid')]
        #        
        #        # At least one edge
        #        clause = sdd.false() if len(edges) > 0 else sdd.true()
        #        for edge in edges:
        #            old_clause = clause
        #            clause = clause | edge
        #            clause.ref()
        #            old_clause.deref()

        #        old_alpha = alpha
        #        alpha = alpha & clause
        #        alpha.ref()
        #        old_alpha.deref()

        #        # At most k out of n variables is true
        #        beta = sdd.true()
        #        for comb in itertools.combinations(edges, 2):
        #            clause = (-comb[0] | -comb[1])
        #            clause.ref()
    
        #            beta_old = beta
        #            beta = beta & clause
        #            beta.ref()
        #            beta_old.deref()

        #        old_alpha = alpha
        #        alpha = alpha & beta
        #        alpha.ref()
        #        old_alpha.deref()

        #import pdb; pdb.set_trace()
