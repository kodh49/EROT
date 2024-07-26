# Required modules
import os
import sys
import torch
import jax
import argparse
import warnings
import jax.numpy as jnp
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

def _cartesian_product_chunked(n, N, start, end):
    ranges = [jnp.arange(n)] * N
    grid = jnp.meshgrid(*ranges, indexing='ij')
    product = jnp.stack(grid, axis=-1).reshape(-1, N)
    return product[start:end]

def get_marginals_to_update(N: int, k: int):
    choice = jax.random.choice(jax.random.PRNGKey(0), N, (k,), replace=False)
    return choice

def single_strong_coulomb_cost(index, marginals_to_update):
    total_cost = 0
    for i in marginals_to_update:
        for j in marginals_to_update:
            if i < j:
                diff = jnp.abs(index[i] - index[j])
                total_cost += jnp.where(diff != 0, 2 / diff, jnp.inf)
    return total_cost

def single_weak_coulomb_cost(index, marginals_to_update):
    total_cost = 0
    for i in marginals_to_update:
        for j in marginals_to_update:
            if i < j:
                diff = jnp.abs(index[i] - index[j])
                total_cost += jnp.where(diff != 0, 2 / diff, 1e+8)
    return total_cost

def single_euclidean_cost(index, marginals_to_update):
    total_cost = 0
    for i in marginals_to_update:
        for j in marginals_to_update:
            if i < j:
                diff = jnp.abs(index[i] - index[j])
                total_cost += jnp.where(diff != 0, diff**2, 0)
    return total_cost

def compute_cost(n, N, single_cost, initial_batch_size=None, k=3):
    shape = (n,) * N
    total_indices = n ** N

    if initial_batch_size is None:
        initial_batch_size = total_indices // 10  # Adjust the initial batch size to a reasonable value

    C = jnp.zeros(shape)
    compute_cost_vmap = jax.vmap(single_cost, in_axes=(0, None))

    current_batch_size = initial_batch_size
    for batch_start in range(0, total_indices, current_batch_size):
        try:
            batch_end = min(batch_start + current_batch_size, total_indices)
            batch_indices = _cartesian_product_chunked(n, N, batch_start, batch_end)
            costs = compute_cost_vmap(batch_indices, get_marginals_to_update(N,k))
            C = C.at[tuple(batch_indices.T)].set(costs)
        except ValueError:
            # Reduce batch size on memory exhaustion and retry
            current_batch_size = max(1, current_batch_size // 2)
            logger.warning(f"Memory exhausted. Reducing batch size to {current_batch_size}")
            batch_start -= current_batch_size  # Retry the same batch with a smaller size
    return C

    
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
    
    match cost_type:
        case "euclidean":
            result = compute_cost(n, N, single_cost=single_euclidean_cost)
        case "weak coulomb":
            result = compute_cost(n, N, single_cost=single_weak_coulomb_cost)
        case "strong coulomb":
            result = compute_cost(n, N, single_cost=single_strong_coulomb_cost)
    
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