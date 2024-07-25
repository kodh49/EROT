# Required modules
import os
import ot
import sys
import jax
import torch
import argparse
import warnings
import numpy as np
import jax.numpy as jnp
from functools import lru_cache
from pathlib import Path
from loguru import logger

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

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

# Euclidean cost computes the Euclidean distance between two coordinates
def compute_euclidean_cost_pot(n):
    """
    Computes the Euclidean cost tensor for 2 marginals setting using POT and PyTorch
    """
    # Initialize cost matrix
    x = torch.arange(n, dtype=torch.float32) # vector in \R^n of the form [1,...,n]
    C = ot.dist(x.reshape((n,1)), x.reshape((n,1))) # Euclidean metric as a cost function
    return C/C.max() # normalize the cost

# Weak Coulomb cost sets relatively large real value for diagonal entries
def compute_weak_coulomb_cost_pot(n):
    """
    Computes the Weak Coulomb cost tensor for 2 marginals setting using POT and PyTorch
    """
    x = np.arange(n, dtype=np.float32) # vector in \R^n of the form [1,...,n]
    # L1 metric with diagonal entries of 1s
    C = torch.from_numpy(ot.dist(x.reshape((n,1)), x.reshape((n,1)), metric='cityblock')) + torch.diag(torch.ones(n))
    C = torch.pow(C,-1) + torch.diag((n+1)*torch.ones(n)) # element-wise inverse and take extreme values for diagonal entries
    return C # normalize the cost

# Strong Coulomb cost sets diagonal entires to be positive infinity
def compute_strong_coulomb_cost_pot(n):
    """
    Computes the Strong Coulomb cost tensor for 2 marginals setting using POT and PyTorch
    """
    x = np.arange(n, dtype=np.float32) # vector in \R^n of the form [1,...,n]
    # L1 metric with diagonal entries of 1s
    C = torch.from_numpy(ot.dist(x.reshape((n,1)), x.reshape((n,1)), metric='cityblock')) + torch.diag(torch.ones(n))
    C = torch.pow(C,-1) + torch.diag(torch.ones(n) * float('inf')) # element-wise inverse and take + infinity for diagonal entries
    return C

@lru_cache(maxsize=None)
def cartesian_product_jax(n, N):
    ranges = [jnp.arange(n)] * N
    grid = jax.numpy.meshgrid(*ranges, indexing='ij')
    product = jax.numpy.stack(grid, axis=-1).reshape(-1, N)
    return product

@lru_cache(maxsize=None)
def cartesian_product_jax_slice(n, N, start, end):
    product = cartesian_product_jax(n, N)
    return product[start:end]

def compute_cost_single(index, N, k, key):
    total_cost = 0
    marginals_to_update = jax.random.choice(key, N, (k,), replace=False)
    for i in marginals_to_update:
        for j in marginals_to_update:
            if i < j:
                diff = jnp.abs(index[i] - index[j])
                total_cost += jnp.where(diff != 0, diff ** 2, jnp.inf)
    return total_cost

def compute_cost_vmap(index_batch, N, k, key):
    return jax.vmap(compute_cost_single, in_axes=(0, None, None, None))(index_batch, N, k, key)

@lru_cache(maxsize=None)
def compute_coulomb_cost_jax(n, N, batch_size=None, k=None):
    """
    Computes the Strong Coulomb cost tensor for multi marginals (N > 2) setting using JAX
    """
    shape = (n,) * N
    indices = cartesian_product_jax(n, N)
    if batch_size is None:
        batch_size = len(indices)
    C = jnp.zeros(shape)
    compute_cost_vmap_fn = jax.vmap(compute_cost_single, in_axes=(0, None, None, None))
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        key = jax.random.PRNGKey(batch_start)  # Create a new key for each batch
        costs = compute_cost_vmap_fn(batch_indices, N, k, key)
        for idx, cost in zip(batch_indices, costs):
            C = C.at[tuple(idx)].set(cost)
    return C

def compute_euclidean_cost_jax(n, N, batch_size=1000, k=2):
    """
    Computes the Euclidean cost tensor for multi marginals (N > 2) setting using JAX
    """
    key = jax.random.PRNGKey(0)
    total_elements = n ** N
    num_batches = total_elements // batch_size + int(total_elements % batch_size != 0)
    C = jnp.zeros((n,) * N)
    compute_cost_vmap_fn = jax.vmap(compute_cost_single, in_axes=(0, None, None, None))
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_elements)
        batch_indices = cartesian_product_jax_slice(n, N, start, end)
        costs = compute_cost_vmap_fn(batch_indices, N, k, key)
        for idx, cost in zip(batch_indices, costs):
            C = C.at[tuple(idx)].set(cost)
    return C

def compute_euclidean_cost(n, N):
    """
    Computes n^N dimensional Euclidean cost tensor for N marginals
    """
    if N == 2:
        return compute_euclidean_cost_pot(n)
    else:
        return compute_euclidean_cost_jax(n, N)
    
def compute_strong_coulomb_cost(n: int, N: int):
    """
    Computes n^N dimensional Strong Coulomb cost tensor for N marginals
    """
    if N == 2:
        return compute_strong_coulomb_cost_pot(n)
    else:
        return compute_coulomb_cost_jax(n, N)

def compute_weak_coulomb_cost(n, N):
    """
    Computes n^N dimensional Weak Coulomb cost tensor for N marginals
    """
    if N == 2:
        return compute_weak_coulomb_cost_pot(n)
    else:
        return compute_coulomb_cost_jax(n, N)
    
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
    
    # generate marginal probability tensor
    logger.info("Generating cost tensor.")
    match cost_type:
        case "euclidean":
            result = compute_euclidean_cost(n, N)
        case "weak coulomb":
            result = compute_weak_coulomb_cost(n, N)
        case "strong coulomb":
            result = compute_strong_coulomb_cost(n, N)
    
    # save generated vector
    logger.info(f"Saving results to {outdir}.")
    torch.save(result, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script generates a cost tensor specified by the provided cost function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)