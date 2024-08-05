# list of external dependencies
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import trange
from pathlib import Path
from loguru import logger
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set JAX environment variables
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"
# Set environment variables to use alql 128 CPU cores
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=128'
os.environ['OMP_NUM_THREADS'] = '128'

import jax
import jax.numpy as jnp