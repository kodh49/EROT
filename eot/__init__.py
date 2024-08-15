import argparse
import sys

from . import (
    make_cost_tensor,
    make_marginal_tensor,
    run_classical_eot,
    run_quantum_eot,
)

def main():
    parser = argparse.ArgumentParser(prog="eot", add_help=False)
    subparsers = parser.add_subparsers(dest="command")

    # Run command with submodules
    run_parser = subparsers.add_parser("run", description="Run the EOT algorithm")
    run_subparsers = run_parser.add_subparsers(dest="run subcommand")

    # Run classical OT algorithms
    classical_parser = run_subparsers.add_parser(
        "classical", description="Run Classical Entropy Regularized Optimal Transport Algorithms."
    )
    run_classical_eot.add_arguments(classical_parser)
    classical_parser.set_defaults(func=run_classical_eot.main)

    # Run quantum OT algorithms
    quantum_parser = run_subparsers.add_parser(
        "quantum", description="Run Quantum Entropy Regularized Optimal Transport Algorithms."
    )
    run_quantum_eot.add_arguments(quantum_parser)
    quantum_parser.set_defaults(func=run_quantum_eot.main)


    

    # Create command with submodules
    create_parser = subparsers.add_parser(
        "create",
        description="Create cost tensors or marginal distribution tensors to be used in EOT algorithms.",
    )
    create_subparsers = create_parser.add_subparsers(dest="create_subcommand")

    # Create cost tensor
    cost_parser = create_subparsers.add_parser(
        "cost", description="Create a cost tensor."
    )
    make_cost_tensor.add_arguments(cost_parser)
    cost_parser.set_defaults(func=make_cost_tensor.main)

    # Create marginal distribution tensor
    marginal_parser = create_subparsers.add_parser(
        "marginal", description="Create a marginal distribution tensor."
    )
    make_marginal_tensor.add_arguments(marginal_parser)
    marginal_parser.set_defaults(func=make_marginal_tensor.main)

    args = parser.parse_args()
    
    if "func" in args:
        args.func(args)


"""
if __name__ == "__main__":
    main()
"""