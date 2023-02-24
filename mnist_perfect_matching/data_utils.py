import os
from functools import partial
from itertools import product, count
from comb_modules.perfect_matching import perfect_matching
from comb_modules.utils import edges_from_grid
from mnist_perfect_matching.edge_costs import edge_cost_fns
from utils import TrainingIterator, LazyTrainingIterator, TransformTrainingIterator,TransformLazyTrainingIterator, load_pickle
import numpy as np
from decorators import input_to_numpy
from PIL import Image
import ray

names = ["train", "val", "test", "val_extra", "test_extra"]
filename_ends = ("full_images", "board", "perfect_matching")

import sys

def thumbnail(img_np, dim, grayscale):
    # convert to grayscale
    im = Image.fromarray(np.uint8(img_np))
    if grayscale:
        im = im.convert('L')
    im.thumbnail((dim, dim), Image.ANTIALIAS)
    im.load()
    return np.asarray(im, dtype=np.uint8) if not grayscale else np.asarray(im, dtype=np.uint8)[:,:,None]


def load_mnist_digits(digits_to_use, thumbnail_size, grayscale):
    train_data, test_data = load_pickle("/is/rg/al/Data/comb_gradient/mnist/original_data/pickled_mnist.pkl")
    images = np.broadcast_to(np.reshape(train_data[:digits_to_use, 1:], newshape=(-1, 28, 28, 1)), (digits_to_use, 28, 28, 3))

    thumbnails = np.asarray([thumbnail(digit, thumbnail_size, grayscale) for digit in images])
    labels = np.reshape(train_data[:digits_to_use, :1], newshape=(-1))
    return thumbnails, labels


def generate_graph(board, board_edges, edge_cost_fn):
    costs = np.zeros(len(board_edges))
    graph_edges = np.zeros((len(board_edges), 2)).astype(np.int32)
    for i, (x, y, xn, yn) in enumerate(board_edges):
        costs[i] = edge_cost_fn(board[x, y], board[xn, yn])
        graph_edges[i] = [x * len(board) + y, xn * len(board) + yn]
    return graph_edges, costs.astype(np.float32)


def generate_board(board_dim, grayscale, pixels_per_cell, digits, labels, edge_cost_fn, enforce_uniqueness):
    edges = edges_from_grid(board_dim, neighbourhood_fn="4-grid")
    board = np.zeros((board_dim, board_dim))
    img = np.zeros((pixels_per_cell * board_dim, pixels_per_cell * board_dim, 1 if grayscale else 3))
    if not grayscale:
        img[:, :] = [0.0, 100.0, 0.0]
    digit_idxs = np.random.choice(list(range(len(digits))), board_dim * board_dim)
    ij_coordinates = product(range(board_dim), range(board_dim))
    for (i, j), digit_idx in zip(ij_coordinates, digit_idxs):
        sprite = digits[digit_idx]
        offset_x = int((pixels_per_cell - sprite.shape[0]) / 2)
        offset_y = int((pixels_per_cell - sprite.shape[1]) / 2)
        img[
            i * pixels_per_cell + offset_x : i * pixels_per_cell + sprite.shape[0] + offset_x,
            j * pixels_per_cell + offset_y : j * pixels_per_cell + sprite.shape[1] + offset_y,
        ] = sprite
        board[i, j] = labels[digit_idx]

    V = board.shape[0] ** 2
    graph_edges, costs = generate_graph(board, edges, edge_cost_fn)
    PM = perfect_matching(graph_edges.tolist(), costs, V)

    if enforce_uniqueness:
        for i in range(3):
            costs_perturbed = costs + 0.01*np.random.normal(size=costs.shape)
            PM_perturbed = perfect_matching(graph_edges.tolist(), costs_perturbed, V)
            if not np.allclose(PM_perturbed, PM):
                return None

    assert len(img.shape) == 3
    assert img.shape[-1] == 1 and grayscale or img.shape[-1]==3 and not grayscale, f"Images shape is {img.shape[-1]}"

    channels_first = img.transpose((2, 0, 1))

    return channels_first, board, PM


def generate_perfect_matching_boards(
    board_dim, grayscale, num_examples, pixels_per_cell, thumbnail_size, edge_cost_fn_name, mnist_digits_to_use, enforce_uniqueness
):
    digits, labels = load_mnist_digits(mnist_digits_to_use, thumbnail_size, grayscale=grayscale)
    edge_cost_fn = edge_cost_fns[edge_cost_fn_name]

    #@ray.remote
    #def parallel():
    while True:
        res = generate_board(
            board_dim, grayscale=grayscale, pixels_per_cell=pixels_per_cell, digits=digits, labels=labels,
            edge_cost_fn=edge_cost_fn, enforce_uniqueness=enforce_uniqueness)
        if res is not None:
            return res
    # generate 500 at a time
    board_chunk_size = 500
    examples = []
    print(f"Generating {num_examples} on {board_dim}x{board_dim} board...")
    for c in range(int(num_examples/board_chunk_size)):
        examples.extend(ray.get([parallel.remote() for _ in range(board_chunk_size)]))
    return examples


def generate_dataset(
    save_dir,
    pixels_per_cell,
    board_dim,
    grayscale,
    thumbnail_size,
    enforce_uniqueness,
    edge_cost_fn_name,
    mnist_digits_to_use,
    train_set_params,
    val_set_params,
    test_set_params,
    extra_val_set_params=None,
    extra_test_set_params=None,
):
    sys.setrecursionlimit(50000)
    generate_grids = partial(
        generate_perfect_matching_boards,
        pixels_per_cell=pixels_per_cell,
        board_dim=board_dim,
        thumbnail_size=thumbnail_size,
        edge_cost_fn_name=edge_cost_fn_name,
        mnist_digits_to_use=mnist_digits_to_use,
        enforce_uniqueness=enforce_uniqueness,
        grayscale=grayscale
    )

    train_set = generate_grids(**train_set_params)
    val_set = generate_grids(**val_set_params)
    test_set = generate_grids(**test_set_params)

    extra_val_set, extra_test_set = None, None
    if extra_val_set_params:
        extra_val_set = generate_grids(**extra_val_set_params)

    if extra_test_set_params:
        extra_test_set = generate_grids(**extra_test_set_params)

    datasets = [train_set, val_set, test_set, extra_val_set, extra_test_set]
    for dataset, name in zip(datasets, names):
        if not dataset:
            continue
        as_four_arrays = zip(*dataset)
        chunk_dim_limit = 8 if not grayscale else 32
        for arr, filename_end in zip(as_four_arrays, filename_ends):
            if board_dim > chunk_dim_limit and "train" in name:
                # save in N parts
                N = int(board_dim / chunk_dim_limit) + 1
                part_size = int(len(arr) / N)
                print(f"Saving in {N} parts!")
                for i, idx in enumerate(range(0, len(arr), part_size)):
                    arr_part = arr[idx : idx + part_size]
                    final_name = name + "_" + filename_end + f"_part{i}.npy"
                    np.save(os.path.join(save_dir, final_name), arr_part)
            else:
                final_name = name + "_" + filename_end + ".npy"
                np.save(os.path.join(save_dir, final_name), arr)


def load_dataset(data_dir, use_test_set, evaluate_with_extra, normalize):
    train_prefix = "train"
    data_suffix = "full_images"
    true_weights_suffix = ""

    local_paths = ["/mnt/local-fast/data/", "/localdata/data/", "/home/mvlastelica/data"]

    val_prefix = ("test" if use_test_set else "val") + ("_extra" if evaluate_with_extra else "")
    import os
    for lp in local_paths:
        local_path = data_dir.replace("/is/rg/al/Data/", lp)
        print(f"Testing {local_path}")
        if os.path.exists(local_path):
            data_dir = local_path
            print("Using local path!")
            break

    if os.path.exists(os.path.join(data_dir, train_prefix + "_" + data_suffix + ".npy")):
        train_inputs = np.load(os.path.join(data_dir, train_prefix + "_" + data_suffix + ".npy"))
        train_labels = np.load(os.path.join(data_dir, train_prefix + "_perfect_matching.npy"))
        train_true_weights =  np.load(os.path.join(data_dir, train_prefix + "_board.npy"))
        if normalize:
            mean, std = (
                np.mean(train_inputs, axis=(0, 2, 3), keepdims=True),
                np.std(train_inputs, axis=(0, 2, 3), keepdims=True),
            )
            train_inputs -= mean
            train_inputs /= std
        train_iterator = TrainingIterator(dict(images=train_inputs, labels=train_labels, true_weights=train_true_weights))
    else:
        input_base_path = os.path.join(data_dir, train_prefix + "_" + data_suffix)
        label_base_path = os.path.join(data_dir, train_prefix + f"_perfect_matching")
        true_weights_base_path = os.path.join(data_dir, train_prefix + f"_board")
        if normalize:
            mean = 128 if not "grayscale" in input_base_path else 0.5
            std =  128 if not "grayscale" in input_base_path else 0.5
            train_iterator = TransformLazyTrainingIterator(dict(images=input_base_path, labels=label_base_path, true_weights=true_weights_base_path), dict(images=lambda x: (x-mean)/std))
        else:
            train_iterator = LazyTrainingIterator((dict(images=input_base_path, labels=label_base_path, true_weights=true_weights_base_path)))

    val_inputs = np.load(os.path.join(data_dir, val_prefix + "_" + data_suffix + ".npy"))

    if normalize:
        val_inputs -= mean
        val_inputs /= std

    val_labels = np.load(os.path.join(data_dir, val_prefix + "_perfect_matching.npy"))
    val_true_weights = np.load(os.path.join(data_dir, val_prefix + "_board.npy"))
    eval_iterator = TrainingIterator(
        dict(images=val_inputs, labels=val_labels, full_images=val_inputs, true_weights=val_true_weights)
    )
    @input_to_numpy
    def denormalize(x):
         return (x*std)+mean
    metadata = {
        "input_image_size": val_inputs[0].shape[-1],
        "output_features": val_true_weights[0].shape[-1] * val_true_weights[0].shape[-2],
        "num_channels": val_inputs[0].shape[0],
        "denormalize": denormalize
    }

    return train_iterator, eval_iterator, metadata
