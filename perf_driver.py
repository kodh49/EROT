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

# Set up a local cluster
cluster = LocalCluster()
client = Client(cluster)

# import local developments
from eot.make_cost_tensor import *
from eot.make_marginal_tensor import *

x = jnp.linspace(-5, 5, 100)

# Marginal distributions
mu_1 = compute_gaussian_marginal(-5, 5, 100, [1.5], [0.5])
mu_2 = compute_gaussian_marginal(-5, 5, 100, [2.5], [0.9])
mu_3 = compute_gaussian_marginal(-5, 5, 100, [2], [0.6])
mu_4 = compute_gaussian_marginal(-5, 5, 100, [3], [0.8])
mu_5 = compute_gaussian_marginal(-5, 5, 100, [2.75], [0.7])

# Strong Coulomb Cost
strong_coulomb_2m = compute_cost(100, 2, single_cost=single_strong_coulomb_cost)
strong_coulomb_3m = compute_cost(100, 3, single_cost=single_strong_coulomb_cost)
strong_coulomb_4m = compute_cost(100, 4, single_cost=single_strong_coulomb_cost)

# Weak Coulomb Cost
weak_coulomb_2m = compute_cost(100, 2, single_cost=single_weak_coulomb_cost)
weak_coulomb_3m = compute_cost(100, 3, single_cost=single_weak_coulomb_cost)
weak_coulomb_4m = compute_cost(100, 4, single_cost=single_weak_coulomb_cost)

# Quadratic Cost
quadratic_2m = compute_cost(100, 2, single_cost=single_euclidean_cost)
quadratic_3m = compute_cost(100, 3, single_cost=single_euclidean_cost)
quadratic_4m = compute_cost(100, 4, single_cost=single_euclidean_cost)

@jax.jit
def remove_tensor_sum(c, u):
    k = len(u)
    for i in range(k):
        c -= jnp.expand_dims(u[i], axis=list(range(i)) + list(range(i + 1, k)))
    return c

@jax.jit
def coupling_tensor(potentials, cost_t, epsilon) -> jnp.ndarray:
    return jnp.exp(-remove_tensor_sum(cost_t, potentials) / epsilon)

def tensor_marginal(coupling: jnp.ndarray, slice_index: int) -> jnp.ndarray:
    k = coupling.ndim
    axis = list(range(slice_index)) + list(range(slice_index + 1, k))
    return coupling.sum(axis=axis)

@jax.jit
def tensor_marginals(tensor):
    return tuple(tensor_marginal(tensor, ix) for ix in range(tensor.ndim))

@jax.jit
def compute_error(potentials, marginals, cost, epsilon):
    coupl_tensor = coupling_tensor(potentials, cost, epsilon)
    computed_marginals = tensor_marginals(coupl_tensor)
    errors = jnp.array([
        jnp.sum(jnp.abs(marginal - computed_marginal))
        for marginal, computed_marginal in zip(marginals, computed_marginals)
    ])
    return errors

def sinkhorn_logsumexp(marginals, c, reg, precision=1e-5, max_iters=20000):
    """
    Solves the multimarginal optimal transport problem using the Sinkhorn algorithm.

    Args:
        marginals (list of jnp.ndarray): List of N marginal distributions, each of shape (n,).
        c (jnp.ndarray): Cost tensor of shape (n, n, ..., n) for N dimensions..
        reg (float): Regularization parameter.
        precision (float, optional): Desired precision for convergence. Default is 1e-4.
        max_iters (int, optional): Maximum number of iterations. Default is 10000.

    Returns:
        tuple: A tuple containing:
            - P (jnp.ndarray): Computed joint probability distribution of shape (n, n, ..., n).
            - log_data (dict): Log data containing the number of steps, time taken, and errors.
    """
    start_time = time.time()
    N = len(marginals)
    n = marginals[0].shape[0]

    potentials = [jnp.zeros(n) for i in range(N)]

    def body_fn(var):
        (potentials, iter, errors) = var
        
        errors = jax.lax.cond(
            iter % 10 == 0,
            lambda: compute_error(potentials, marginals, c, reg),
            lambda: errors
        )
        
        iter += 1
        
        for i in range(N):
            axis = list(range(i)) + list(range(i + 1, N))
            lse = logsumexp(remove_tensor_sum(c, potentials) / -reg, axis=axis)
            potentials[i] += reg * jnp.log(marginals[i]) - reg * lse
        return potentials, iter, errors

    potentials, iter, errors = jax.lax.while_loop(
        lambda var: jnp.logical_and(jnp.max(var[2]) > precision, var[1] <= max_iters),
        lambda var: body_fn(var),
        (potentials, 0, jnp.full(N, jnp.inf))
    )

    P = coupling_tensor(potentials, c, reg)

    log_data = {
        'steps': iter,
        'time': time.time() - start_time,
        'errors': errors,
    }

    return P, log_data

# Define the regularization parameters, marginals, and cost matrices
regularization = [10, 1, 0.5, 1, 1e-1, 1e-2, 1e-3, 1e-4]
marginals = [mu_1, mu_2, mu_3, mu_4]
weak_coulomb_costs = [weak_coulomb_2m, weak_coulomb_3m, weak_coulomb_4m]
strong_coulomb_costs = [strong_coulomb_2m, strong_coulomb_3m, strong_coulomb_4m]
quadratic_costs = [quadratic_2m, quadratic_3m, quadratic_4m]

# Create empty DataFrames for each case
df_2marginals = pd.DataFrame()
df_3marginals = pd.DataFrame()
df_4marginals = pd.DataFrame()

def run_sinkhorn_for_config(marginals_subset, cost, reg, cost_type, marginals_count):
    # Run Sinkhorn algorithm (assuming you have this function defined as per your code)
    P, log_data = sinkhorn_logsumexp(marginals_subset, cost, reg)
    
    return {
        'regularization': reg,
        'cost_type': cost_type,
        'marginals': marginals_count,
        'steps': log_data['steps'],
        'time': log_data['time'],
        'errors': log_data['errors'].tolist()
    }

def compute_marginal_results(marginals, regularization, weak_coulomb_costs, strong_coulomb_costs, quadratic_costs):
    results = []
    
    # Loop over the number of marginals
    for i in range(2, 5):
        marginals_subset = marginals[:i]
        if i == 2:
            cost_matrices = [(weak_coulomb_costs[0], 'Weak Coulomb'), (strong_coulomb_costs[0], 'Strong Coulomb'), (quadratic_costs[0], 'Quadratic')]
        elif i == 3:
            cost_matrices = [(weak_coulomb_costs[1], 'Weak Coulomb'), (strong_coulomb_costs[1], 'Strong Coulomb'), (quadratic_costs[1], 'Quadratic')]
        elif i == 4:
            cost_matrices = [(weak_coulomb_costs[2], 'Weak Coulomb'), (strong_coulomb_costs[2], 'Strong Coulomb'), (quadratic_costs[2], 'Quadratic')]
        
        # Parallel computation for each combination of regularization and cost matrix
        results.extend([
            run_sinkhorn_for_config(marginals_subset, cost, reg, cost_type, i)
            for reg in regularization
            for cost, cost_type in cost_matrices
        ])

    return results

if __name__ == '__main__':
    # Step 1: Set up a local Dask cluster using all available CPUs and cores automatically
    cluster = LocalCluster()
    client = Client(cluster)

    # Optional: print out cluster information
    print(client)
    dask.freeze_support()

    # Step 3: Run the computation with Dask
    results = compute_marginal_results(marginals, regularization, weak_coulomb_costs, strong_coulomb_costs, quadratic_costs)

    # Step 4: Trigger the computation and gather results
    computed_results = dask.compute(*results)
    logger.success("Computation complete")

    # Step 5: Convert the results list into a DataFrame for each marginal case
    df_2marginals = pd.DataFrame([result for result in computed_results if result['marginals'] == 2])
    df_3marginals = pd.DataFrame([result for result in computed_results if result['marginals'] == 3])
    df_4marginals = pd.DataFrame([result for result in computed_results if result['marginals'] == 4])

    logger.success("Saving Dataframes")

    # Save DataFrames to files if needed
    df_2marginals.to_csv('df_2marginals.csv', index=False)
    df_3marginals.to_csv('df_3marginals.csv', index=False)
    df_4marginals.to_csv('df_4marginals.csv', index=False)