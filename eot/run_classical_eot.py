from ..env.dependencies import *
import classical_eot as classical
import utils as utils

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)


def add_arguments(parser):
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


def main(args):
    entropy = args.entropy # type of entropy regularizer
    cost_tensor_path = str(Path(args.cost).absolute())  # path to cost.pt file
    marginal_path = args.marginal  # list of paths to marginal.pt files
    epsilon = args.epsilon  # Minimum probability of individual true negative.
    num_iter = args.num_iter  # Number of iterations for alternating maximization.
    error = args.error  # Maximum convergence error of the algorithm
    out = str(Path(args.out).absolute()) # full path to output pytorch tensor file
    outdir = os.path.join(os.path.dirname(out), Path(out).stem)  # path to output directory
    out_filename = os.path.basename(out)  # output filename
    plotdir = os.path.join(outdir, "plots") # path to plot directory


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
    logger.info("Loading the cost tensor file generated.")
    cost = torch.load(cost_tensor_path)

    # load marginal tensors
    logger.info("Loading marginal tensor files.")
    marg = list(map(torch.load, marginal_path))
    
    # Run the computation based on specification
    match entropy:
        case "shannon":
            logger.info("Computing Shannon regularized Classical OT.")
            algs = {
                "Sinkhorn": classical.shannon_sinkhorn
                # listed algorithms will be executed
            }
        case "quadratic":
            logger.info("Computing Quadratic regularized Classical OT.")
            algs = {
                "Cyclic Projection": classical.quadratic_cyclic_projection,
                "Fixed Point iteration": classical.quadratic_fixed_point_iteration,
                "Gradient Descent": classical.quadratic_gradient_descent,
                "Nesterov Gradient Descent": classical.quadratic_nesterov_gradient_descent
            }

    result = {}
    # multiprocessing
    with mp.Manager() as manager:
        result_queue = manager.Queue() # create a joint queue
        processes = []
        i = 0
        for func_label, func in algs.items():
            gpu_device_num = i % torch.cuda.device_count()
            process = mp.Process(target=utils.mp_add_queue, args=(
                result_queue,
                func_label,
                func,
                cost, marg, epsilon, gpu_device_num, num_iter, error
            ))
            processes.append(process)
            process.start()
            i += 1
        
        for process in processes:
            process.join()
    
        while not result_queue.empty():
            key, value = result_queue.get()
            result[key] = value

    # save plots of results
    logger.info(f"Saving plots to {plotdir}.")
    # utils.plot_matrices({'Cost Matrix': cost}, os.path.join(plotdir, "cost_plot"))
    utils.plot_matrices(result, os.path.join(plotdir, "coupling_plots"))

    # save results to an pytorch tensor file
    logger.info(f"Saving coupling tensors to {outdir}.")
    torch.save(result, os.path.join(outdir, "tensors", out_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script computes classical optimal n-couplings from multiple marginal distributions and a cost tensor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    mp.set_start_method('spawn') # define multiprocessing context
    main(args)