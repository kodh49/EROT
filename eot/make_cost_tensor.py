# Required modules
import os
import sys
import utils 
import torch

# Set the environment variable before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import time
from tqdm import trange
import argparse
import warnings
from pathlib import Path
from loguru import logger

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

device = utils.select_gpu()

def cartesian_product_jax(n, N):
    ranges = [jnp.arange(n)] * N
    grid = jnp.meshgrid(*ranges, indexing='ij')
    product = jnp.stack(grid, axis=-1).reshape(-1, N)
    return product

def single_strong_coulomb_cost(index, N, k):
    total_cost = 0
    marginals_to_update = jax.random.choice(jax.random.PRNGKey(0), N, (k,), replace=False)
    for i in marginals_to_update:
        for j in marginals_to_update:
            if i < j:
                diff = jnp.abs(index[i] - index[j])
                total_cost += jnp.where(diff != 0, 2 / diff, jnp.inf)
    return total_cost

def single_weak_coulomb_cost(index, N, k):
    total_cost = 0
    marginals_to_update = jax.random.choice(jax.random.PRNGKey(0), N, (k,), replace=False)
    for i in marginals_to_update:
        for j in marginals_to_update:
            if i < j:
                diff = jnp.abs(index[i] - index[j])
                total_cost += jnp.where(diff != 0, 2 / diff, 1e+8)
    return total_cost

def single_euclidean_cost(index, N, k):
    total_cost = 0
    marginals_to_update = jax.random.choice(jax.random.PRNGKey(0), N, (k,), replace=False)
    for i in marginals_to_update:
        for j in marginals_to_update:
            if i < j:
                diff = jnp.abs(index[i] - index[j])
                total_cost += jnp.where(diff != 0, diff**2, jnp.inf)
    return total_cost

def compute_cost(n: int, N: int, single_cost, batch_size = None, k = 2):
    shape = (n,) * N
    indices = cartesian_product_jax(n, N)
    if batch_size is None:
        batch_size = len(indices) // 10
    C = jnp.zeros(shape)
    compute_cost_vmap = jax.vmap(single_cost, in_axes=(0, None, None))
    for batch_start in trange(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        costs = compute_cost_vmap(batch_indices, N, k)
        C = C.at[tuple(batch_indices.T)].set(costs)
    return C

def add_arguments(parser):
    parser.add_argument(
        "--n",
        type=int,
        help="Number of data points of the cost tensor.",
        required=True,
    )
    parser.add_argument(
        "--N",
        type=int,
        help="Number of marginals",
        required=True,
        default=2
    )
    parser.add_argument(
        "--cost_type",
        type=str,
        choices=["euclidean", "weak coulomb", "strong coulomb"],
        help="Governing equation for the cost.",
        required=False,
        default=[0]
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to output tensor.",
        required=False,
        default=os.path.join(os.getcwd(), "mu.pt"),
    )
    
def main(args):
    n = args.n # number of points in each space
    N = args.N # number of marginals
    cost_type = args.cost_type
    out = str(Path(args.out).absolute())
    outdir = os.path.dirname(out)
    out_filename = os.path.basename(out)

    # check if the output filename is valid
    if os.path.splitext(out_filename)[1] != ".pt":
        raise ValueError(
            f"Output filename {out} is not a valid pytorch tensor file. Please use .pt as the extension."
        )
    
    # Make sure the output can be written to
    if not os.access(outdir, os.W_OK):
        # give error message, and exit with error status
        print(f"Cannot write to the location: {outdir}.\n")
        print("Please check if this location exists, and that you have the permission to write to this location. Exiting..\n")
        sys.exit(1)
    
    start_time = time.time()
    match cost_type:
        case "euclidean":
            result = compute_cost(n, N, single_cost=single_euclidean_cost)
        case "weak coulomb":
            result = compute_cost(n, N, single_cost=single_weak_coulomb_cost)
        case "strong coulomb":
            result = compute_cost(n, N, single_cost=single_strong_coulomb_cost)
    end_time = time.time()
    # save generated vector
    logger.info(f"Elapsed Time {end_time-start_time}")
    logger.info(f"Saving results to {outdir}.")
    result = torch.from_dlpack(jax.dlpack.to_dlpack(result))
    torch.save(result, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script generates a cost tensor specified by the provided cost function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)