# import all necessary external dependencies
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from env.lib import *

x = jnp.linspace(-5, 5, 100)

mu_1 = norm.pdf(x, loc=1.5, scale=0.5)
mu_2 = norm.pdf(x, loc=2.5, scale=0.9)
mu_3 = norm.pdf(x, loc=2, scale=0.6)
mu_4 = norm.pdf(x, loc=3, scale=0.8)
mu_5 = norm.pdf(x, loc=2.75, scale=0.7)

mu_1 = mu_1 / mu_1.sum()
mu_2 = mu_2 / mu_2.sum()
mu_3 = mu_3 / mu_3.sum()
mu_4  = mu_4 / mu_4.sum()
mu_5 = mu_5 / mu_5.sum()

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_strong_coulomb_jax(x, N):
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns)

    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, jnp.inf)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)

    return total_cost

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_weak_coulomb_jax(x, N):
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns)

    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, 1e+8)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)

    return total_cost

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_quadratic_jax(x, N):
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns)

    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, diff**2, 0)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)

    return total_cost


strong_coulomb_2m = compute_cost_matrix_strong_coulomb_jax(x,2)
strong_coulomb_3m = compute_cost_matrix_strong_coulomb_jax(x,3)
strong_coulomb_4m = compute_cost_matrix_strong_coulomb_jax(x,4)

weak_coulomb_2m = compute_cost_matrix_weak_coulomb_jax(x,2)
weak_coulomb_3m = compute_cost_matrix_weak_coulomb_jax(x,3)
weak_coulomb_4m = compute_cost_matrix_weak_coulomb_jax(x,4)

quadratic_2m = compute_cost_matrix_quadratic_jax(x,2)
quadratic_3m = compute_cost_matrix_quadratic_jax(x,3)
quadratic_4m = compute_cost_matrix_quadratic_jax(x,4)


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

def sinkhorn_logsumexp(marginals: list, c: jnp.ndarray, reg: float, precision: float = 1e-5, max_iters=20000):
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
regularization = [10, 1, 0.5, 1, 1e-1, 1e-2, 1e-3]
marginals = [mu_1, mu_2, mu_3, mu_4]
weak_coulomb_costs = [weak_coulomb_2m, weak_coulomb_3m, weak_coulomb_4m]
strong_coulomb_costs = [strong_coulomb_2m, strong_coulomb_3m, strong_coulomb_4m]
quadratic_costs = [quadratic_2m, quadratic_3m, quadratic_4m]

# Create empty DataFrames for each case
df_2marginals = pd.DataFrame()
df_3marginals = pd.DataFrame()
df_4marginals = pd.DataFrame()

# Loop over the number of marginals
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for i in range(2, 5):
        marginals_subset = marginals[:i]
        if i == 2:
            cost_matrices = [(weak_coulomb_costs[0], strong_coulomb_costs[0], quadratic_costs[0])]
        elif i == 3:
            cost_matrices = [(weak_coulomb_costs[1], strong_coulomb_costs[1], quadratic_costs[1])]
        elif i == 4:
            cost_matrices = [(weak_coulomb_costs[2], strong_coulomb_costs[2], quadratic_costs[2])]
        
        results = []
        for reg in regularization:
            for strong_coulomb_cost, weak_coulomb_cost, quadratic_cost in cost_matrices:
                # Run Sinkhorn algorithm for Coulomb cost
                P_coulomb, log_data_coulomb = sinkhorn_logsumexp(marginals_subset, strong_coulomb_cost, reg)
                results.append({
                    'regularization': reg,
                    'cost_type': 'Strong Coulomb',
                    'marginals': i,
                    'steps': log_data_coulomb['steps'],
                    'time': log_data_coulomb['time'],
                    'errors': log_data_coulomb['errors'].tolist()
                })
                logger.success(f"Complete {i}-marginal Sinkhorn for Strong Coulomb regularization {reg}")

                # Run Sinkhorn algorithm for Coulomb cost
                P_coulomb, log_data_coulomb = sinkhorn_logsumexp(marginals_subset, weak_coulomb_cost, reg)
                results.append({
                    'regularization': reg,
                    'cost_type': 'Weak Coulomb',
                    'marginals': i,
                    'steps': log_data_coulomb['steps'],
                    'time': log_data_coulomb['time'],
                    'errors': log_data_coulomb['errors'].tolist()
                })
                logger.success(f"Complete {i}-marginal Sinkhorn for Weak Coulomb regularization {reg}")

                # Run Sinkhorn algorithm for Quadratic cost
                P_quadratic, log_data_quadratic = sinkhorn_logsumexp(marginals_subset, quadratic_cost, reg)
                results.append({
                    'regularization': reg,
                    'cost_type': 'Quadratic',
                    'marginals': i,
                    'steps': log_data_quadratic['steps'],
                    'time': log_data_quadratic['time'],
                    'errors': log_data_quadratic['errors'].tolist()
                })
                logger.success(f"Complete {i}-marginal Sinkhorn for Quadratic regularization {reg}")

        # Convert the results list into a DataFrame and store it in the corresponding DataFrame
        if i == 2:
            df_2marginals = pd.DataFrame(results)
        elif i == 3:
            df_3marginals = pd.DataFrame(results)
        elif i == 4:
            df_4marginals = pd.DataFrame(results)

logger.success("Saving DataFrames")

# Save DataFrames to files if needed
df_2marginals.to_csv('df_2marginals.csv', index=False)
df_3marginals.to_csv('df_3marginals.csv', index=False)
df_4marginals.to_csv('df_4marginals.csv', index=False)