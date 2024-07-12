# Required modules
import os
import ot
import sys
import torch
import argparse
import warnings
import numpy as np
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
def compute_euclidean_cost(n):
    # Initialize cost matrix
    x = torch.arange(n, dtype=torch.float32) # vector in \R^n of the form [1,...,n]
    C = ot.dist(x.reshape((n,1)), x.reshape((n,1))) # Euclidean metric as a cost function
    return C/C.max() # normalize the cost

# Weak Coulomb cost sets relatively large real value for diagonal entries
def compute_weak_coulomb_cost(n):
    x = np.arange(n, dtype=np.float32) # vector in \R^n of the form [1,...,n]
    # L1 metric with diagonal entries of 1s
    C = torch.from_numpy(ot.dist(x.reshape((n,1)), x.reshape((n,1)), metric='cityblock')) + torch.diag(torch.ones(n))
    C = torch.pow(C,-1) + torch.diag((n**3+1)*torch.ones(n)) # element-wise inverse and take extreme values for diagonal entries
    return C/C.max() # normalize the cost

# Strong Coulomb cost sets diagonal entires to be positive infinity
def compute_strong_coulomb_cost(n):
    x = np.arange(n, dtype=np.float32) # vector in \R^n of the form [1,...,n]
    # L1 metric with diagonal entries of 1s
    C = torch.from_numpy(ot.dist(x.reshape((n,1)), x.reshape((n,1)), metric='cityblock')) + torch.diag(torch.ones(n))
    C = torch.pow(C,-1) + torch.diag(torch.ones(n) * float('inf')) # element-wise inverse and take + infinity for diagonal entries
    return C

def main(args):
    n = args.n
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
            result = compute_euclidean_cost(n)
        case "weak coulomb":
            result = compute_weak_coulomb_cost(n)
        case "strong coulomb":
            result = compute_strong_coulomb_cost(n)
    
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
