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