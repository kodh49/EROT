# list of external dependencies
import os
import sys
import time
import warnings
from loguru import logger
from pathlib import Path
from tqdm import trange
import argparse
from functools import partial

# Set the environment variable before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"
# Set environment variables to use alql 128 CPU cores
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=128'
os.environ['OMP_NUM_THREADS'] = '128'

import jax
import jax.numpy as jnp

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

@partial(jax.jit, static_argnums=(0, 1))
def cartesian_product_jax(n: jnp.int64, N: jnp.int64):
    ranges = [jnp.arange(n, dtype=jnp.int64)] * N
    grid = jnp.meshgrid(*ranges, indexing='ij')
    product = jnp.stack(grid, axis=-1).reshape(-1, N)
    return product

@partial(jax.jit, static_argnums=1)
def single_strong_coulomb_cost(index, N: jnp.uint64):
    """
    Computes the Strong Coulomb cost for indexed marginals
    """
    marginals_to_update = jax.random.choice(jax.random.PRNGKey(0), N, (2,), replace=False)
    i, j = marginals_to_update
    diff = jnp.abs(index[i] - index[j])
    return jnp.where(diff != 0, 2 / diff, jnp.inf)

@partial(jax.jit, static_argnums=1)
def single_weak_coulomb_cost(index, N: jnp.uint64):
    """
    Computes the Weak Coulomb cost for indexed marginals
    """
    marginals_to_update = jax.random.choice(jax.random.PRNGKey(0), N, (2,), replace=False)
    i, j = marginals_to_update
    diff = jnp.abs(index[i] - index[j])
    return jnp.where(diff != 0, 2 / diff, 1e+8)

@partial(jax.jit, static_argnums=1)
def single_euclidean_cost(index, N: jnp.uint64):
    """
    Computes the Euclidean cost for indexed marginals
    """
    marginals_to_update = jax.random.choice(jax.random.PRNGKey(0), N, (2,), replace=False)
    i, j = marginals_to_update
    diff = jnp.abs(index[i] - index[j])
    return jnp.where(diff != 0, diff**2, jnp.inf)

def compute_cost(n: jnp.uint64, N: jnp.uint64, single_cost, batch_size: jnp.uint64 = None):
    """
    Computes the cost tensor of N marginal probability vectors that are n-discretized under specified single_cost function
    """
    start_time = time.time()
    shape = (n,) * N
    logger.info("Generating cartesian product")
    indices = cartesian_product_jax(n, N)
    if batch_size is None:
        batch_size = jnp.uint64(len(indices) // 10)
    logger.info("Initializing cost tensor")
    C = jnp.zeros(shape)
    compute_cost_vmap = jax.vmap(single_cost, in_axes=(0, None))
    logger.info("Starting batch operations")
    for batch_start in trange(0, len(indices), batch_size):
        batch_indices = indices[batch_start:jnp.uint64(batch_start + batch_size)]
        costs = compute_cost_vmap(batch_indices, N)
        C = C.at[tuple(batch_indices.T)].set(costs)
    end_time = time.time()
    return C, end_time - start_time


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
        default=os.path.join(os.getcwd(), "cost"),
    )
    
def main(args):
    n = args.n # number of points in each space
    N = args.N # number of marginals
    cost_type = args.cost_type # type of the cost function
    out = str(Path(args.out).absolute()) # file save metadata
    outdir = os.path.dirname(out)
    
    # Make sure the output can be written to
    if not os.access(outdir, os.W_OK):
        # give error message, and exit with error status
        print(f"Cannot write to the location: {outdir}.\n")
        print("Please check if this location exists, and that you have the permission to write to this location. Exiting..\n")
        sys.exit(1)
    
    match cost_type:
        case "euclidean":
            result, elapsed_time = compute_cost(n, N, single_cost=single_euclidean_cost)
        case "weak coulomb":
            result, elapsed_time = compute_cost(n, N, single_cost=single_weak_coulomb_cost)
        case "strong coulomb":
            result, elapsed_time = compute_cost(n, N, single_cost=single_strong_coulomb_cost)
    
    logger.info(f"Elapsed Time {elapsed_time}")
    logger.info(f"Saving results to {outdir}.")
    jnp.save(out, result) # save generated tensor into .npy format

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script generates a cost tensor specified by the provided cost function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)