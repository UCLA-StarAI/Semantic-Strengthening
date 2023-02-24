
from mnist_perfect_matching import data_utils
from comb_modules.utils import edges_from_grid
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

def perfect_matching_vis(grid_img, grid_dim, labels, color=(0,255,255), width=2):

    edges = edges_from_grid(grid_dim, neighbourhood_fn='4-grid')
    pixels_per_cell = int(grid_img.shape[0]/grid_dim)

    img = Image.fromarray(np.uint8(grid_img.squeeze()), mode='RGB')
    for i, (y1,x1, y2, x2) in enumerate(edges):
        if labels[i]:
            draw = ImageDraw.Draw(img)
            draw.line((x1*pixels_per_cell + pixels_per_cell/2, y1*pixels_per_cell + pixels_per_cell/2,
            x2*pixels_per_cell + pixels_per_cell/2, y2*pixels_per_cell + pixels_per_cell/2), fill=color,width=width)
            del draw

    return np.asarray(img, dtype=np.uint8)
