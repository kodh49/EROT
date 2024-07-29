# Required modules
import os
import sys
import argparse
import warnings
from scipy.stats import norm
from pathlib import Path
from loguru import logger

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

def add_arguments(parser):
    parser.add_argument(
        "--n",
        type=int,
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
        default=os.path.join(os.getcwd(), "mu"),
    )

# generate marginal probability vector in \R^n supported on [lend, rend]
# resulting vector is a linear combination of normal distributions with means=locs and standard deviations=scales
def compute_gaussian_marginal(lend, rend, n, locs, scales):
    x = jnp.linspace(lend, rend, n)
    mu = jnp.zeros(n)
    for (loc, scale) in zip(locs, scales):
        mu += (norm.pdf(x,loc=loc, scale=scale))/len(locs)
    mu = mu / mu.sum()
    return mu

# we may support additional distributions

def main(args):
    n = args.n
    locs = args.locs
    scales = args.scales
    out = str(Path(args.out).absolute())
    outdir = os.path.dirname(out)
    out_filename = os.path.basename(out)
    
    # Make sure the output can be written to
    if not os.access(outdir, os.W_OK):
        # give error message, and exit with error status
        print(f"Cannot write to the location: {outdir}.\n")
        print("Please check if this location exists, and that you have the permission to write to this location. Exiting..\n")
        sys.exit(1)
    
    # generate marginal probability tensor
    logger.info("Generating probability vector.")
    result = compute_gaussian_marginal(-5, 5, n, locs, scales)
    jnp.save(out, result) # save generated probability vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script generates marginal distributions as a linear combination of multiple normal distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
