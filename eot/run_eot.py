import os
import sys
import torch
import classical_eot as classical
import quantum_eot as quantum
from pathlib import Path
import argparse
import utils as utils
import warnings
from loguru import logger

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)


def add_arguments(parser):
    parser.add_argument(
        "--ot",
        type=str,
        choices=["classical", "quantum"],
        help="Type of optimal transport problem",
        required=True,
    )
    parser.add_argument(
        "--entropy",
        type=str,
        choices=["quadratic", "shannon"],
        help="Type of entropy regularizer.",
        required=True,
    )
    parser.add_argument(
        "--cost",
        type=str,
        help="Path to a .pt file containing the pytorch cost tensor.",
        required=True,
    )
    parser.add_argument(
        "--marginal",
        nargs="+",
        type=str,
        help="A list of paths to each .pt file containing a marginal probability vector. "
        "Each probability vector will be used in listed order.",
        required=True,
        default=[os.path.join(os.getcwd(),"/tests/testdata/mu_1.pt"), os.path.join(os.getcwd(),"/tests/testdata/mu_2.pt")],
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Regularization factor.",
        required=True,
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        help="Number of alternating maximization iterations.",
        required=False,
        default=50000,
    )
    parser.add_argument(
        "--error",
        type=float,
        help="Maximum convergence error.",
        required=False,
        default=1e-8,
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to output tensor.",
        required=False,
        default=os.path.join(os.getcwd(), "result.pt"),
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        help="Device number of the GPU that will be used for computation.",
        required=False,
        default=0,
    )


def main(args):
    ot_type = args.ot # type of optimal transport problem
    entropy = args.entropy # type of entropy regularizer
    cost_tensor_path = str(Path(args.cost).absolute())  # path to cost.pt file
    marginal_path = args.marginal  # list of paths to marginal.pt files
    epsilon = args.epsilon  # Minimum probability of individual true negative.
    num_iter = args.num_iter  # Number of iterations for alternating maximization.
    error = args.error  # Maximum convergence error of the algorithm
    gpu_device_num = args.gpu_device # GPU device number
    out = str(Path(args.out).absolute())  # full path to output pytorch tensor file
    outdir = os.path.dirname(out)  # path to output directory
    out_filename = os.path.basename(out)  # output filename

    # check if the output filename is valid
    if os.path.splitext(out_filename)[1] != ".pt":
        raise ValueError(
            f"Output filename {out} is not a valid pytorch tensor file. Please use .pt as the extension."
        )

    # check if the cost tensor file exists
    utils.check_file_existence(
        cost_tensor_path,
        f"Cost tensor file {cost_tensor_path} does not exist. "
        f"Please run make_cost_tensor.py first.",
    )

    # check if marginal files exist
    for x in marginal_path:
        utils.check_file_existence(
            x,
            f"Marginal tensor file {x} does not exists. "
            f"Please run make_marginal_tensor.py first.",
        )

    # check if selected gpu is available for computation
    if not (0 <= gpu_device_num < torch.cuda.device_count()):
        raise ValueError(
            f"Selected GPU device cuda:{gpu_device_num} is not available for computation. "
        )

    # Make sure the output can be written to
    if not os.access(outdir, os.W_OK):
        # give error message, and exit with error status
        print(f"Cannot write to the location: {outdir}.\n")
        print("Please check if this location exists, and that you have the permission to write to this location. Exiting..\n")
        sys.exit(1)

    # load the cost tensor
    logger.info("Loading the cost tensor file generated.")
    cost = torch.load(cost_tensor_path)

    # load marginal tensors
    logger.info("Loading marginal tensor files.")
    marg = list(map(torch.load, marginal_path))

    # check that the cost tensor has the same dimension as marginal vectors specified
    
    
    # Run the computation based on specification
    result = {}
    match (ot_type, entropy):
        case ("classical", "shannon"):
            logger.info("Computing Shannon regularized Classical OT.")
        case ("classical", "quadratic"):
            logger.info("Computing Quadratic regularized Classical OT.")
            result["cyclic projection"] = classical.quadratic_cyclic_projection(
                C=cost, marg=marg, epsilon=epsilon, gpu=gpu_device_num, num_iter=num_iter, convergence_error=error
            )
            result["fixed point iteration"] = classical.quadratic_fixed_point_iteration(
                C=cost, marg=marg, epsilon=epsilon, gpu=gpu_device_num, num_iter=num_iter, convergence_error=error
            )
            result["gradient descent"] = classical.quadratic_gradient_descent(
                C=cost, marg=marg, epsilon=epsilon, gpu=gpu_device_num, num_iter=num_iter, convergence_error=error
            )
            result["nesterov gradient descent"] = classical.quadratic_nesterov_gradient_descent(
                C=cost, marg=marg, epsilon=epsilon, gpu=gpu_device_num, num_iter=num_iter, convergence_error=error
            )
        case ("quantum", "von neumann"):
            logger.info("Computing Von Neumann regularized Quantum OT.")
        case ("quantum", "quadratic"):
            logger.info("Computing Quadratic regularized Quantum OT.")

    # process the results and save them to an pytorch tensor file
    logger.info(f"Saving results to {outdir}.")
    torch.save(result, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script computes optimal n-couplings from multiple marginal distributions and a cost tensor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
