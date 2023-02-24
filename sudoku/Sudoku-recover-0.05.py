#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import argparse
from collections import namedtuple

import numpy as np
import numpy.random as npr

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex, clear_output

import sys
sys.path.insert(0,'/space/ahmedk/approxback/circuit_utils')
sys.path.insert(0,'/space/ahmedk/approxback')
from logexp import *
from circuit import *
from recover_sudoku import *

import time

# torch.use_deterministic_algorithms(True)
torch.manual_seed(4)

import random
random.seed(4)

import numpy as np
np.random.seed(4)


class FigLogger(object):
    def __init__(self, fig, base_ax, title):
        self.colors = ['tab:red', 'tab:blue']
        self.labels = ['Loss (entropy)', 'Error']
        self.markers = ['d', '.']
        self.axes = [base_ax, base_ax.twinx()]
        base_ax.set_xlabel('Epochs')
        base_ax.set_title(title)
        
        for i, ax in enumerate(self.axes):
            ax.set_ylabel(self.labels[i], color=self.colors[i])
            ax.tick_params(axis='y', labelcolor=self.colors[i])

        self.reset()
        self.fig = fig
        
    def log(self, args):
        for i, arg in enumerate(args[-2:]):
            self.curves[i].append(arg)
            x = list(range(len(self.curves[i])))
            self.axes[i].plot(x, self.curves[i], self.colors[i], marker=self.markers[i])
            self.axes[i].set_ylim(0, 3.05)
            
        self.fig.canvas.draw()
        
    def reset(self):
        for ax in self.axes:
            for line in ax.lines:
                line.remove()
        self.curves = [[], []]

import pysdd
import itertools
from pysdd.sdd import SddManager, Vtree
import itertools

INTERVAL = 10
K = 2
since_recover = 0


# Constraints per row/col/square
import pysdd
import itertools
from pysdd.sdd import SddManager, Vtree
import itertools

sdd = pysdd.sdd.SddManager(var_count=9*9*9, auto_gc_and_minimize=True)

def cell(x, y, z):
    return sdd.vars[(z + y*9 + x*81) + 1]
N = 9

#unique row
row_constraints = []
for row in range(N):
    print(f'{row+1}/9')
    constraint = sdd.true()
    for val in range(N):
        for col in range(N - 1):
            for col_inner in range(col + 1, N):
                constraint_old = constraint
                constraint = constraint & (-cell(row, col, val) | -cell(row, col_inner, val))
                constraint.ref()
                constraint_old.deref()
    sdd.minimize_limited()
    row_constraints.append(constraint)
    
# unique col
col_constraints = []
for col in range(N):
    print(f'{col+1}/9')
    constraint = sdd.true()
    for val in range(N):
        for row in range(N - 1):
            for row_inner in range(row + 1, N):
                constraint_old = constraint
                constraint = constraint & (-cell(row, col, val) | -cell(row_inner, col, val))
                constraint.ref()
                constraint_old.deref()
    sdd.minimize_limited()
    col_constraints.append(constraint)

# Every cell has to be unique in its current square
block_constraints = []
for block_row, block_col in itertools.product(range(0, 9, 3), range(0, 9, 3)):
    print('block')
    constraint = sdd.true()
    cells = list(itertools.product(range(block_row, block_row+3), range(block_col, block_col+3)))
    for val in range(N):
        for i in range(8):
            for j in range(i+1, 9):
                constraint_old = constraint
                constraint = constraint & (-cell(cells[i][0], cells[i][1], val) | -cell(cells[j][0], cells[j][1], val))
                constraint.ref()
                constraint_old.deref()
    sdd.minimize_limited()
    block_constraints.append(constraint)
    
constraints = row_constraints + col_constraints + block_constraints

import time
def run(boardSz, epoch, model, optimizer, logger, dataset, batchSz, to_train=False, unperm=None):

    # Recover constraints
    global since_recover
    if  since_recover % INTERVAL == 0 and epoch != 0: #and self.last_batch_time < 1.2:

        model.eval()
        recover_iterator = iter(DataLoader(dataset, shuffle=True, batch_size=1000)) 
        data, is_input, label = next(recover_iterator)

        with torch.no_grad():
            print("Start recovering constraints....")

            global constraints
            constraints = recover(constraints, data.view(-1,9,9,9), model, is_input, K=5, var_map=var_map)
            print("End recovery... " + str(len(constraints)) + " remaining.")

            print("Minimizing SDD size...")
            sdd.garbage_collect()
            sdd.minimize_limited()
            print("Done with minimization.")

    since_recover+=1
    
    if to_train:
        model.train()
    else:
        model.eval()
    
    loss_final, err_final = 0, 0

    loader = DataLoader(dataset, shuffle=True, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))

    for i,(data, is_input, label) in tloader:
        
        is_target = 1 - is_input
        is_input = is_input.view(-1, 9, 9, 9)
        label = label.view(-1 ,9,9,9)
        if to_train: optimizer.zero_grad()
        preds = model(data.view(-1, 9, 9, 9))
        is_target = 1 - data.view(-1, 9,9,9).sum(3).flatten()
        celoss = nn.functional.cross_entropy(preds.view(-1, 9), label.view(-1, 9,9,9).argmax(-1).flatten(), reduction='none')
        celoss = (celoss * is_target).mean()
        
        # Semantic Loss
        output = F.softmax(preds.view(-1, 9), dim=-1).view(-1, 9*9*9)
        output = torch.where(is_input.bool().view(batchSz, -1), torch.tensor(0.).cuda(), output)
        output = torch.where(data == 1,  torch.tensor(1.).cuda(), output)
        outputu = torch.unbind(output, axis=1)
        
        lit_weights = [[1-p, p] for p in outputu]
        sl = torch.stack([(-wmc(c, lit_weights, log_space=False).log()).mean() for c in constraints])
        sloss = 0.05 * torch.sum(sl)
        
        loss = celoss + sloss

        if to_train:
            loss.backward()
            optimizer.step()
            
        err = computeErr(preds.data, boardSz, unperm, data, is_input)/len(data)
        
        tloader.set_description('Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err))
        loss_final += loss.item()
        err_final += err

    loss_final, err_final = loss_final/len(loader), err_final/len(loader)
    logger.log((epoch, loss_final, err_final))

    if not to_train:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_final, err_final))


def train(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, True, unperm)

@torch.no_grad()
def test(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, False, unperm)
    
@torch.no_grad()
def computeErr(pred_flat, n, unperm, data, is_input):
    if unperm is not None: pred_flat[:,:] = pred_flat[:,unperm]

    nsq = n ** 2
    pred = pred_flat.view(-1, nsq, nsq, nsq)

    batchSz = pred.size(0)
    s = (nsq-1)*nsq//2 # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)
    I = torch.where(data.view(-1, 9,9,9).sum(3).bool(), data.view(-1,9,9,9).argmax(3), I)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq-1)
        valid *= (x.sum(1) == s)
        return ~valid

    boardCorrect = torch.ones(batchSz).type_as(pred)
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:,j,:])] = 0
        boardCorrect[invalidGroups(I[:,:,j])] = 0

        # Check the jth block.
        row, col = n*(j // n), n*(j % n)
        M = invalidGroups(I[:,row:row+n,col:col+n].contiguous().view(batchSz,-1))
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return float(batchSz-boardCorrect.sum())


class ConvNet(nn.Module):
    def __init__(self, boardSz):
        super().__init__()

        self.boardSz = boardSz

        convs = []
        Nsq = boardSz**2
        prevSz = Nsq
        szs = [512]*10 + [Nsq]
        for sz in szs:
            conv = nn.Conv2d(prevSz, sz, kernel_size=3, padding=1)
            convs.append(conv)
            prevSz = sz

        self.convs = nn.ModuleList(convs)

    def __call__(self, x):
        nBatch = x.size(0)
        Nsq = x.size(1)
        mask = x.view(-1, 9,9,9).sum(3)
        for i in range(len(self.convs)-1):
            x = F.tanh(self.convs[i](x))
        x = self.convs[-1](x)
        return x
    
def process_inputs(X, Ximg, Y, boardSz):
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()

    Ximg = Ximg.flatten(start_dim=1, end_dim=2)
    Ximg = Ximg.unsqueeze(2).float()

    X      = X.view(X.size(0), -1)
    Y      = Y.view(Y.size(0), -1)
    is_input = is_input.view(is_input.size(0), -1)

    return X, Ximg, Y, is_input

with open('sudoku/features.pt', 'rb') as f:
    X_in = torch.load(f)
with open('sudoku/features_img.pt', 'rb') as f:
    Ximg_in = torch.load(f)
with open('sudoku/labels.pt', 'rb') as f:
    Y_in = torch.load(f)
with open('sudoku/perm.pt', 'rb') as f:
    perm = torch.load(f)
    
X_in = Y_in.clone().detach()
X_in = X_in.view(-1, 81, 9)
tmp = torch.randint(0, 81, (Y_in.shape[0], 17))
X_in[torch.arange(10000).unsqueeze(1), tmp] = torch.tensor([0.]*9)
X_in = X_in.view(-1, 9,9,9)

X, Ximg, Y, is_input = process_inputs(X_in, Ximg_in, Y_in, 3)
X, Ximg, is_input, Y = X.cuda(), Ximg.cuda(), is_input.cuda(), Y.cuda()

N = X_in.size(0)
nTrain = int(N*0.9)

sudoku_train = TensorDataset(X[:nTrain], is_input[:nTrain], Y[:nTrain])
sudoku_test =  TensorDataset(X[nTrain:], is_input[nTrain:], Y[nTrain:])


def show_sudoku(raw):
    return (torch.argmax(raw,2)+1)*(raw.sum(2).long())

def show_mnist_sudoku(raw):
    A = raw.numpy()
    digits = np.concatenate(np.concatenate(A,axis=1), axis=1).astype(np.uint8)
    linewidth = 2
    board = np.zeros((digits.shape[0]+linewidth*4, digits.shape[1]+linewidth*4), dtype=np.uint8)
    gridwidth = digits.shape[0]//3

    board[:] = 255
    for i in range(3):
        for j in range(3):
            xoff = linewidth+(linewidth+gridwidth)*i
            yoff = linewidth+(linewidth+gridwidth)*j
            xst = gridwidth*i
            yst = gridwidth*j
            board[xoff:xoff+gridwidth, yoff:yoff+gridwidth] = digits[xst:xst+gridwidth, yst:yst+gridwidth]

    plt.imshow(255-digits, cmap='gray')

display(Markdown('## Sudoku'))
print(show_sudoku(X_in[0]))
print()
display(Markdown('## One-hot encoded Boolean Sudoku'))
print(X[0])
    
display(Markdown('## MNIST Sudoku'))


get_ipython().run_cell_magic('capture', '', "mnist_sudoku = ConvNet(3).cuda()\noptimizer = optim.Adam([{'params': mnist_sudoku.parameters(), 'lr': 5e-4}])\n# \n# optimizer = optim.SGD([{'params': mnist_sudoku.parameters(), 'lr': 0.1}])\n\n# optimizer = torch.optim.SGD(mnist_sudoku.parameters(), lr=0.1, momentum=0.9)\n\nfig, axes = plt.subplots(1,2, figsize=(10,4))\nplt.subplots_adjust(wspace=0.4)\ntrain_logger = FigLogger(fig, axes[0], 'Traininig')\ntest_logger = FigLogger(fig, axes[1], 'Testing')")


batch_size = 128
test(3, 0, mnist_sudoku, optimizer, test_logger, sudoku_test, 100)
for epoch in range(1, 100+1):
    train(3, epoch, mnist_sudoku, optimizer, train_logger, sudoku_train, 100)
    test(3, epoch, mnist_sudoku, optimizer, test_logger, sudoku_test, 100)
    display(fig)
