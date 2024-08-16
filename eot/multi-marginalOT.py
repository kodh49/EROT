import logging
from scipy.stats import norm
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
import jax
from jax.scipy.special import logsumexp
import time
import pandas as pd
from dask import delayed, compute
from dask.distributed import Client

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

# Set up Dask client to use the local scheduler
client = Client(n_workers=128)
logging.info(f'Dask client set up with {client.ncores()} cores.')

# Function to compute the Coulomb cost matrix using JAX
@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_coulomb_jax(x, N):
    logging.info(f'Starting computation of Coulomb cost matrix for {N} marginals')
    
    n = x.shape[0]
    ns = (n,) * N  # Create a tuple with N elements, each of size n
    total_cost = jnp.zeros(ns)  # Initialize the total cost tensor with zeros

    # Define the Coulomb pairwise cost function
    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, jnp.inf)  # Avoid division by zero by returning infinity

    # Loop over all pairs of marginals to compute pairwise costs
    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)  # Add the computed cost to the total cost tensor

    logging.info(f'Finished computation of Coulomb cost matrix for {N} marginals')
    return total_cost

# Function to compute the Quadratic cost matrix using JAX
@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_quadratic_jax(x, N):
    logging.info(f'Starting computation of Quadratic cost matrix for {N} marginals')
    
    n = x.shape[0]
    ns = (n,) * N  # Create a tuple with N elements, each of size n
    total_cost = jnp.zeros(ns)  # Initialize the total cost tensor with zeros

    # Define the Quadratic pairwise cost function
    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, diff**2, 0)  # Return squared differences for non-zero differences

    # Loop over all pairs of marginals to compute pairwise costs
    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)  # Add the computed cost to the total cost tensor

    logging.info(f'Finished computation of Quadratic cost matrix for {N} marginals')
    return total_cost

# Function to remove the tensor sum, used for Sinkhorn iterations
@jax.jit
def remove_tensor_sum(c, u):
    k = len(u)
    for i in range(k):
        c -= jnp.expand_dims(u[i], axis=list(range(i)) + list(range(i + 1, k)))  # Subtract potential u from cost
    return c

# Function to compute the coupling tensor from the potentials and cost matrix
@jax.jit
def coupling_tensor(potentials, cost_t, epsilon) -> jnp.ndarray:
    return jnp.exp(-remove_tensor_sum(cost_t, potentials) / epsilon)  # Exponential to get the coupling tensor

# Function to compute the marginal of the coupling tensor along a given axis
def tensor_marginal(coupling: jnp.ndarray, slice_index: int) -> jnp.ndarray:
    k = coupling.ndim
    axis = list(range(slice_index)) + list(range(slice_index + 1, k))
    return coupling.sum(axis=axis)  # Sum over all axes except the one corresponding to the marginal

# Function to compute all marginals of the coupling tensor
@jax.jit
def tensor_marginals(tensor):
    return tuple(tensor_marginal(tensor, ix) for ix in range(tensor.ndim))

# Function to compute the error between the target and computed marginals
@jax.jit
def compute_error(potentials, marginals, cost, epsilon):
    coupl_tensor = coupling_tensor(potentials, cost, epsilon)
    computed_marginals = tensor_marginals(coupl_tensor)
    errors = jnp.array([
        jnp.sum(jnp.abs(marginal - computed_marginal))
        for marginal, computed_marginal in zip(marginals, computed_marginals)
    ])
    return errors  # Return the array of errors for each marginal

# Implementation of the Sinkhorn algorithm using log-sum-exp stabilization
def sinkhorn_logsumexp(marginals, c, reg, precision=1e-3, max_iters=20000):
    start_time = time.time()
    N = len(marginals)
    n = marginals[0].shape[0]

    logging.info(f'Starting Sinkhorn algorithm with regularization = {reg}, precision = {precision}, max_iters = {max_iters}')

    potentials = [jnp.zeros(n) for i in range(N)]  # Initialize potentials with zeros

    # Function for the body of the while loop in the Sinkhorn algorithm
    def body_fn(var):
        (potentials, iter, errors) = var

        # Compute errors every 200 iterations
        errors = jax.lax.cond(
            iter % 200 == 0,
            lambda: compute_error(potentials, marginals, c, reg),
            lambda: errors
        )

        iter += 1

        # Update each potential using log-sum-exp to ensure numerical stability
        for i in range(N):
            axis = list(range(i)) + list(range(i + 1, N))
            lse = logsumexp(remove_tensor_sum(c, potentials) / -reg, axis=axis)
            potentials[i] += reg * jnp.log(marginals[i]) - reg * lse
        return potentials, iter, errors

    # Run the while loop until the error is below the precision or max iterations are reached
    potentials, iter, errors = jax.lax.while_loop(
        lambda var: jnp.logical_and(jnp.max(var[2]) > precision, var[1] <= max_iters),
        lambda var: body_fn(var),
        (potentials, 0, jnp.full(N, jnp.inf))
    )

    # Compute the final coupling tensor
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

# Define the regularization parameters, marginals, and cost matrices
regularization = [10, 1, 0.5, 1e-01, 1e-02, 1e-03, 1e-04]
marginals = [mu_1, mu_2, mu_3]
coulomb_costs = [delayed(compute_cost_matrix_coulomb_jax)(x, 2), delayed(compute_cost_matrix_coulomb_jax)(x, 3)]
quadratic_costs = [delayed(compute_cost_matrix_quadratic_jax)(x, 2), delayed(compute_cost_matrix_quadratic_jax)(x, 3)]

# Function to run Sinkhorn algorithm for different cases
@delayed
def run_sinkhorn_for_case(cost_fn, marginals, reg):
    cost_matrix = cost_fn.compute()
    logging.info(f'Starting Sinkhorn for regularization {reg} with cost function {cost_fn.__name__}')
    P, log_data = sinkhorn_logsumexp(marginals, cost_matrix, reg)
    logging.info(f'Finished Sinkhorn run for regularization {reg} with cost function {cost_fn.__name__}')
    return {
        'regularization': reg,
        'cost_type': cost_fn.__name__.split('_')[-1].capitalize(),
        'marginals': [marginal.tolist() for marginal in marginals],  # Convert to list for easy logging
        'steps': log_data['steps'],
        'time': log_data['time'],
        'errors': log_data['errors']
    }

# Create tasks for Dask
tasks = []
for reg in regularization:
    for cost_fn in coulomb_costs + quadratic_costs:
        tasks.append(run_sinkhorn_for_case(cost_fn, marginals, reg))

# Compute all tasks in parallel
results = compute(*tasks)

# Process and save results
results_df = pd.DataFrame(results)
results_df.to_csv('sinkhorn_results.csv', index=False)
logging.info('Results have been saved to sinkhorn_results.csv')
