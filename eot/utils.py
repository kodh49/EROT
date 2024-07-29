# list of external dependencies
import os
import sys
import torch
import time
import warnings
import numpy as np
from loguru import logger
from pathlib import Path
from tqdm import trange
import argparse
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

# Set the environment variable before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

# Global color variables
COLORS = ['b','g','r','c','m','y']
HEATMAP_COLOR = plt.cm.inferno

def check_file_existence(file_path: str, error_description: str) -> None:
    """
    Helper function that checks if a file exists. If not, raises a ValueError with the given error description.
    :param file_path: string (location of the file)
    :param error_description: string (description of the error)
    :return: None
    """
    if not os.path.exists(file_path):
        raise ValueError(error_description)

def construct_arguments(vars, i):
    result = []
    for (j, var) in enumerate(vars):
        if j != i:
            result.append(var)
            result.append([j])
    return result

def get_all_arguments(vars):
    result = []
    for (j, var) in enumerate(vars):
        result.append(var)
        result.append([j])
    return result


def plot_vectors(n: int, data: dict, filename: str) -> None:
    """
    Plots vectors supported on [-5,5] with respect to labels.
    """
    x = np.linspace(-5, 5, n)
    fig = plt.figure(figsize=(8, 6))
    i = 0
    for label, vec in data.items():
        plt.plot(x, vec, label=rf'\{label}', color=COLORS[i])
        i += 1
    plt.legend()
    fig.savefig(f"{filename}.png")

# save plots of a dictionary of pytorch tensors with attached labels
def plot_matrices(data: dict, filename: str) -> None:
  labels, matrices = list(data.keys()), list(data.values())
  if len(data) == 1:
      fig, axes = plt.subplots(1, 1, figsize=(14,14))
  elif len(data) == 2:
      fig, axes = plt.subplots(1, 2, figsize=(14,14)) 
  elif len(data) == 3:
      fig, axes = plt.subplots(1, 3, figsize=(14,14))
  elif len(data) == 4:
      fig, axes = plt.subplots(2, 2, figsize=(14,14))
  else:
      logger.error("plot_matrices support 4 tensors at maximum.")
      sys.exit(1)
  # Loop through subplots and plot each matrix
  axes = axes.flatten()
  for i, matrix in enumerate(matrices):
    ax = axes[i]
    im = ax.imshow(matrix, interpolation='nearest', cmap=HEATMAP_COLOR, extent=(0.5, np.shape(matrix)[0] + 0.5, 0.5, np.shape(matrix)[1] + 0.5))
    ax.set_title(labels[i])  # Add title to each subplot (optional)
  # Adjust layout (optional)
  fig.colorbar(im, ax=axes, shrink=0.5, location="bottom")
  fig.savefig(f"{filename}.png")    

# add a function that takes the string and function pointer and add it to a mp.Queue
def mp_add_queue(result_queue, label: str, func, *func_args) -> None:
    obj = func(*func_args)
    result_queue.put((label, obj))