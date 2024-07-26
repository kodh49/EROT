# Required modules
import os
import sys
import utils 
import torch
import jax
import time
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

def _cartesian_product_chunked(n, N, start, end):
    ranges = [torch.arange(n, device=device)] * N
    grid = torch.meshgrid(*ranges, indexing='ij')
    product = torch.stack(grid, dim=-1).reshape(-1, N)
    return product[start:end]

def get_marginals_to_update(N: int, k: int):
    choice = jax.dlpack.to_dlpack(jax.random.choice(jax.random.PRNGKey(0), N, (k,), replace=False))
    return torch.tensor(choice)

def single_strong_coulomb_cost(index, marginals_to_update):
    indices = index[:, marginals_to_update]
    diffs = torch.abs(indices.unsqueeze(2) - indices.unsqueeze(1))
    mask = torch.triu(torch.ones(diffs.shape[-2:], device=device), diagonal=1)
    cost_matrix = torch.where(diffs != 0, 2 / diffs, torch.tensor(float('inf'), device=device))
    total_cost = (cost_matrix * mask).sum(dim=[1, 2])
    return total_cost

def single_weak_coulomb_cost(index, marginals_to_update):
    indices = index[:, marginals_to_update]
    diffs = torch.abs(indices.unsqueeze(2) - indices.unsqueeze(1))
    mask = torch.triu(torch.ones(diffs.shape[-2:], device=device), diagonal=1)
    cost_matrix = torch.where(diffs != 0, 2 / diffs, torch.tensor(1e+8, device=device))
    total_cost = (cost_matrix * mask).sum(dim=[1, 2])
    return total_cost

def single_euclidean_cost(index, marginals_to_update):
    indices = index[:, marginals_to_update]
    diffs = torch.abs(indices.unsqueeze(2) - indices.unsqueeze(1))
    mask = torch.triu(torch.ones(diffs.shape[-2:], device=device), diagonal=1)
    cost_matrix = diffs ** 2
    total_cost = (cost_matrix * mask).sum(dim=[1, 2])
    return total_cost

def compute_cost(n: int, N: int, single_cost, batch_size = None, k = 3) -> torch.Tensor:
    start_time = time.time()
    shape, total_indices = (n,) * N, n ** N
    if batch_size is None:
        batch_size = total_indices // 10  # adjust the initial batch size
    C = torch.zeros(shape, device=device)
    # take adaptive batch sizes
    current_batch_size = batch_size
    for batch_start in range(0, total_indices, current_batch_size):
        try:
            batch_end = min(batch_start + current_batch_size, total_indices)
            batch_indices = _cartesian_product_chunked(n, N, batch_start, batch_end).to(device)
            marginals_to_update = get_marginals_to_update(N, k).to(device)
            costs = single_cost(batch_indices, marginals_to_update)
            C[tuple(batch_indices.T)] = costs
        except RuntimeError: # memory exhaustion
            current_batch_size = max(1, current_batch_size // 2) # reduce batch size
            batch_start -= current_batch_size  # reattempt the same batch with a smaller size
    end_time = time.time()
    logger.success(f"Time taken: {end_time - start_time}")
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