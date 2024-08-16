import logging
import jax.numpy as jnp
import jax 
from jax.scipy.special import logsumexp
import pandas as pd
import time
from scipy.stats import norm
from sinkhorn_functions import (compute_cost_matrix_coulomb_jax, compute_cost_matrix_quadratic_jax,
                                 remove_tensor_sum, coupling_tensor, tensor_marginals, compute_error)
from multiprocessing import Pool

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a range of values for x
x = jnp.linspace(-5, 5, 100)

# Define Gaussian distributions for each marginal
mu_1 = norm.pdf(x, loc=1.5, scale=0.5)
mu_2 = norm.pdf(x, loc=2.5, scale=0.9)
mu_3 = norm.pdf(x, loc=2, scale=0.6)
mu_4 = norm.pdf(x, loc=3, scale=0.8)
mu_5 = norm.pdf(x, loc=2.75, scale=0.7)

# Normalize the distributions so that they sum to 1
mu_1 = mu_1 / mu_1.sum()
mu_2 = mu_2 / mu_2.sum()
mu_3 = mu_3 / mu_3.sum()
mu_4 = mu_4 / mu_4.sum()
mu_5 = mu_5 / mu_5.sum()

# Define the Sinkhorn algorithm
def sinkhorn_logsumexp(marginals, c, reg, precision=1e-3, max_iters=20000):
    start_time = time.time()
    N = len(marginals)
    n = marginals[0].shape[0]

    logging.info(f'Starting Sinkhorn algorithm with regularization = {reg}, precision = {precision}, max_iters = {max_iters}')

    potentials = [jnp.zeros(n) for i in range(N)]

    def body_fn(var):
        (potentials, iter, errors) = var

        errors = jax.lax.cond(
            iter % 200 == 0,
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

    logging.info(f'Sinkhorn algorithm completed in {log_data["steps"]} iterations')
    logging.info(f'Time taken: {log_data["time"]:.2f} seconds')
    logging.info(f'Final errors: {log_data["errors"]}')

    return P, log_data

# Define the function to run Sinkhorn experiments
def run_sinkhorn_experiment(cost_fn, marginals, reg):
    logging.info(f'Starting Sinkhorn for regularization {reg} with cost function {cost_fn.__name__}')
    P, log_data = sinkhorn_logsumexp(marginals, cost_fn, reg)
    logging.info(f'Finished Sinkhorn run for regularization {reg} with cost function {cost_fn.__name__}')
    return {
        'regularization': reg,
        'cost_type': cost_fn.__name__.split('_')[-1].capitalize(),
        'marginals': [marginal.tolist() for marginal in marginals],
        'steps': log_data['steps'],
        'time': log_data['time'],
        'errors': log_data['errors']
    }

# Named functions for cost matrices
def cost_fn_coulomb_2():
    return compute_cost_matrix_coulomb_jax(x, 2)

def cost_fn_coulomb_3():
    return compute_cost_matrix_coulomb_jax(x, 3)

def cost_fn_quadratic_2():
    return compute_cost_matrix_quadratic_jax(x, 2)

def cost_fn_quadratic_3():
    return compute_cost_matrix_quadratic_jax(x, 3)

# Function to parallelize experiments
def parallelize_experiments(cost_functions, marginals, regularization):
    tasks = [(cost_fn, marginals, reg) for cost_fn in cost_functions for reg in regularization]
    
    with Pool() as pool:
        results = pool.starmap(run_sinkhorn_experiment, tasks)
    
    return results

if __name__ == '__main__':
    regularization = [10, 1, 0.5, 1e-01, 1e-02, 1e-03, 1e-04]
    marginals = [mu_1, mu_2, mu_3]
    
    cost_functions = [
        cost_fn_coulomb_2,
        cost_fn_coulomb_3,
        cost_fn_quadratic_2,
        cost_fn_quadratic_3
    ]

    # Parallelize the Sinkhorn experiments
    results = parallelize_experiments(cost_functions, marginals, regularization)

    # Process and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('sinkhorn_results.csv', index=False)
    logging.info('Results have been saved to sinkhorn_results.csv')
