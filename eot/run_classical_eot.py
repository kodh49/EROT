# import all necessary external dependencies
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from env.lib import *

# import internal dependencies
import utils
import classical_eot as classical

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

def add_arguments(parser):
    parser.add_argument(
        "--alg",
        type=str,
        choices=["sinkhorn", "fpi", "cyc-proj" "gradient-descent", "nesterov"],
        help="Type of optimizization method to use: Sinkhorn Algorithm, Fixed Point Iteration, Cyclic Projection, Gradient Descent, and Nesterov Gradient Descent",
        required=True,
    )
    parser.add_argument(
        "--cost",
        type=str,
        help="Path to a .npy file containing the cost tensor.",
        required=True,
    )
    parser.add_argument(
        "--marginal",
        nargs="+",
        type=str,
        help="A list of paths to each .npy file containing a marginal probability vector."
        "Each probability vector will be used in a listed order.",
        required=True,
        default=[os.path.join(os.getcwd(),"/tests/testdata/mu_1.npy"), os.path.join(os.getcwd(),"/tests/testdata/mu_2.npy")],
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Regularization factor of the entropy.",
        required=True,
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        help="Number of optimizing iterations. If not specified, algorithm will continue until it meets convergence error.",
        required=False,
        default = None,
    )
    parser.add_argument(
        "--error",
        type=float,
        help="Maximum convergence error. Set to 1e-6 otherwise specified.",
        required=False,
        default=1e-6,
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to save optimal coupling tensor. The tensor will be saved to <working_dir>/coupling.npy",
        required=False,
        default=os.path.join(os.getcwd(), "coupling.npy"),
    )


def main(args):
    alg = args.alg # type of optimizer
    cost_tensor_path = str(Path(args.cost).absolute())  # path to cost.npy file
    marginal_path = args.marginal  # list of paths to marginal.npy files
    epsilon = args.epsilon  # Minimum probability of individual true negative.
    num_iter = args.num_iter  # Number of iterations for alternating maximization.
    error = args.error  # Maximum convergence error of the algorithm
    out = str(Path(args.out).absolute()) # full path to output pytorch tensor file
    outdir = os.path.join(os.path.dirname(out), Path(out).stem)  # path to output directory
    out_filename = os.path.basename(out)  # output filename
    plotdir = os.path.join(outdir, "plots") # path to plot directory

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

    # Create directory to store results
    logger.info(f"Created directory: {outdir}.\n")
    os.makedirs(os.path.join(outdir, "tensors"))
    os.makedirs(plotdir)
    # Make sure the output can be written to
    if not os.access(outdir, os.W_OK):
        # give error message, and exit with error status
        logger.critical(f"Cannot write to the location: {outdir}.\n Please check if this location exists, and that you have the permission to write to this location. Exiting..\n")
        sys.exit(1)

    # load the cost tensor
    logger.info("Loading the cost tensor.")
    cost = jnp.load(cost_tensor_path)

    # load marginal tensors
    logger.info("Loading marginal distributions.")
    marg = list(map(jnp.load, marginal_path))

    # validate the dimensionality of the problem
    logger.info("Validating dimensionalities.")

    # number of marginals provided matches the rank of the cost tensor
    num_of_marginals = len(marg)
    rank_of_cost = len(cost.shape)
    if num_of_marginals != rank_of_cost:
        logger.error(f"Received {num_of_marginals} marginals while the cost tensor of rank {rank_of_cost} has the shape {cost.shape}.")

    # marginals are 1D vectors
    for marginal, path in zip(marg, marginal_path):
        i = 0
        if len(marginal.shape) != 1:
            logger.error(f"Mismatching dimension: {marginal.shape} \n Please check if the marginal at {path} is a 1 dimensional probability vector. Exiting..\n")
            sys.exit(1)
        
        # gridpoints of marginals matches gridpoints of each dimension of the cost tensor
        if marginal.shape[0] != cost.shape[i]:
            logger.error(f"Mismatching dimension: {marginal.shape[0]} gridpoints are used for marginal at {path}, while {cost.shape[i]} gridpoints are used for the corresponding dimension of the cost tensor. Exiting..\n")
            sys.exit(1)
    
    # Run the computation based on specification
    match alg:
        case "sinkhorn":
            logger.info("Running Multi Marginal Sinkhorn for Shannon Entropy Regularization.")
            algorithm = classical.shannon_sinkhorn

        case "fpi":
            # validity check
            if num_of_marginals != 2:
                logger.critical(f"Cannot run the selected algorithm. \n Fixed Point Iteration only supports 2 marginal settings, while {len(marg)} marginals are provided. Exiting..\n")
                sys.exit(1)
            logger.info("Running Fixed Point Iteration for Quadratic Entropy Regularization.")
            algorithm = classical.quadratic_fixed_point_iteration

        case "cyc-proj":
            # validity check
            if num_of_marginals != 2:
                logger.critical(f"Cannot run the selected algorithm. \n Cyclic Projection only supports 2 marginal settings, while {len(marg)} marginals are provided. Exiting..\n")
                sys.exit(1)
            logger.info("Running Cyclic Projection for Quadratic Entropy Regularization.")
            algorithm = classical.quadratic_cyclic_projection

        case "nesterov":
            # validity check
            if num_of_marginals != 2:
                logger.critical(f"Cannot run the selected algorithm. \n Nesterov Gradient Descent only supports 2 marginal settings, while {len(marg)} marginals are provided. We plan to support multi-marginal setting soon. Exiting..\n")
                sys.exit(1)
            logger.info("Running Nesterov Gradient Descent for Quadratic Entropy Regularization.")
            algorithm = classical.quadratic_nesterov_gradient_descent

        case "gradient-descent":
            # validity check
            if num_of_marginals != 2:
                logger.critical(f"Cannot run the selected algorithm. \n Gradient Descent only supports 2 marginal settings, while {len(marg)} marginals are provided. We plan to support multi-marginal setting soon. Exiting..\n")
                sys.exit(1)
            logger.info("Running Gradient Descent for Quadratic Entropy Regularization.")
            algorithm = classical.quadratic_gradient_descent

    # run selected algorithm
    result, log_data = algorithm(marginals=marg, cost=cost, epsilon=epsilon, convergence_error=error, max_iters=num_iter)

    # save plots of results
    logger.info(f"Saving plots to {plotdir}.")
    utils.plot_matrices(result, os.path.join(plotdir, "coupling_plots"))

    # save the optimal coupling to an npy file
    logger.info(f"Saving coupling tensors to {outdir}.")
    jnp.save(os.path.join(outdir, "tensors", out_filename), result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script solves the classical optimal transport problem under various entropic regularization using multiple available algorithms. It computes the optimal n-coupling from multiple marginal distributions and a cost tensor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)