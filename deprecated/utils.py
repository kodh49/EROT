# utilities including saving and visualizing tensors

# Required modules
import torch
import numpy as np

# save a list of pytorch tensors with a given list of filenames
def save_tensors(matrices, filenames):
    for (matrix, filename) in zip(matrices, filenames):
        torch.save(matrix, f"{filename}.pt")

# load a list of pytorch tensors with a given list of filenames
def load_tensors(filenames):
    for filename in filenames:
        torch.load(f"{filename}.pt")