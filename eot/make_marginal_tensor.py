# Required modules
import os
import sys
import torch
import argparse
import warnings
import numpy as np
from scipy.stats import norm
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
        help="Number of data points of the probability tensor.",
        required=True,
    )
    parser.add_argument(
        "--locs",
        nargs="+",
        type=float,
        help="List of means of normal distributions in [-5,5].",
        required=False,
        default=[0]
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        help="List of standard deviations of normal distributions.",
        required=False,
        default=[1]
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to output tensor.",
        required=False,
        default=os.path.join(os.getcwd(), "mu.pt"),
    )

# generate marginal probability vector in \R^n supported on [lend, rend]
# resulting vector is a linear combination of normal distributions with means=locs and standard deviations=scales
def compute_gaussian_marginal(lend, rend, n, locs, scales):
    x = np.linspace(lend, rend, n)
    mu = np.zeros(n)
    for (loc, scale) in zip(locs, scales):
        mu += (norm.pdf(x,loc=loc, scale=scale))/len(locs)
    mu = torch.from_numpy(mu / mu.sum())
    return mu

# we may support additional distributions

def main(args):
    n = args.n
    locs = args.locs
    scales = args.scales
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
    logger.info("Generating probability vector.")
    result = compute_gaussian_marginal(-5, 5, n, locs, scales)
    
    # save generated vector
    logger.info(f"Saving results to {outdir}.")
    torch.save(result, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script generates marginal distributions as a linear combination of multiple normal distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)