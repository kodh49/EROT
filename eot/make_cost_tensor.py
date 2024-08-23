# import all necessary external dependencies
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from env.lib import *

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

'''
- Purpose:
  These functions compute the cost matrices for MMOT problems,
  where the cost is either based on the Coulomb interaction or a quadratic function.
  The cost matrices are multi-dimensional tensors representing the interaction costs
  between different elements in the marginals.

- Difference:
  The Coulomb cost uses an inverse distance metric (1/distance), 
  while the quadratic cost uses the squared distance (distance^2).

- Advantage: 
  Both are important in different contexts. 
  The Coulomb cost is suitable for physical systems like charged particles, 
  while the quadratic cost is often used in more general optimal transport problems.
'''

@jax.jit
def pairwise_weak_coulomb(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, 1e+8)

@jax.jit
def pairwise_strong_coulumb(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, jnp.inf)

@jax.jit
def pairwise_quadratic(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, diff**2, 0)


@partial(jax.jit, static_argnums=[2])
def compute_cost(pairwise_func, x, N):
    """
    Computes the cost matrix for N marginals under specified pairwise distance function using JAX.

    Args:
        pairwise_func (Callable): function of two variables that computes pairwise distance
        x (jnp.ndarray): Input array of shape (n,).
        N (int): Number of marginals.

    Returns:
        jnp.ndarray: Quadratic cost matrix of shape (n, n, ..., n) for N dimensions.
    """
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns, dtype=jnp.float32)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: pairwise_func(x_, y_))(x))(x)
            '''
            The inner vmap computes a vector of pairwise interactions between a fixed x_ and every other element y_ in x.
            The outer vmap repeats this process for each x_ in x, resulting in a matrix cost_m, where each element represents the pairwise 
            interaction between different elements of x.
            '''
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            '''
            The purpose of this code is to create a list of axis indices that excludes i and j. 
            This is commonly used in operations where you want to sum or manipulate data across all dimensions except for specific ones.
            '''
            total_cost += jnp.expand_dims(cost_m, axis=axis)
            '''
            jnp.expand_dims: Adds new dimensions to cost_m at the specified axis positions.
            axis=axis: Specifies where to insert these new dimensions.
            total_cost += ...: Adds the expanded cost_m to total_cost, leveraging broadcasting to match shapes.
            This operation allows cost_m to be added to total_cost correctly, even if their shapes initially differ, 
            by expanding cost_m to have compatible dimensions.
            '''
    return total_cost


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
        default=os.path.join(os.getcwd(), "cost"),
    )
    
def main(args):
    n = args.n # number of points in each space
    N = args.N # number of marginals
    cost_type = args.cost_type # type of the cost function
    out = str(Path(args.out).absolute()) # file save metadata
    outdir = os.path.dirname(out)
    
    # Make sure the output can be written to
    if not os.access(outdir, os.W_OK):
        # give error message, and exit with error status
        print(f"Cannot write to the location: {outdir}.\n")
        print("Please check if this location exists, and that you have the permission to write to this location. Exiting..\n")
        sys.exit(1)
    
    # select corresponding pairwise distance function and generate the cost tensor
    match cost_type:
        case "quadratic":
            result = compute_cost(pairwise_func=pairwise_quadratic, n=n, N=N)
        case "euclidean":
            result = jnp.sqrt(compute_cost(pairwise_func=pairwise_quadratic, n=n, N=N))
        case "weak coulomb":
            result = compute_cost(pairwise_func=pairwise_weak_coulomb, n=n, N=N)
        case "strong coulomb":
            result = compute_cost(pairwise_func=pairwise_strong_coulumb, n=n, N=N)
    
    logger.info(f"Saving the cost matrix to {outdir}.")
    jnp.save(out, result) # save generated tensor into .npy format

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script generates a cost tensor specified by the provided cost function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)