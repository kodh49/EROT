import logging
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from functools import partial
from scipy.stats import norm
import time
import psutil
import os

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB



@partial(jax.jit, static_argnums=[1, 2])
def compute_cost_matrix_coulomb_jax(x, N, batch_size):
    """
    Computes the Coulomb cost matrix for N marginals using JAX, divided into batches.

    Args:
        x (jnp.ndarray): Input array of shape (n,).
        N (int): Number of marginals.
        batch_size (int): Size of the batch for processing.

    Returns:
        jnp.ndarray: Coulomb cost matrix of shape (n, n, ..., n) for N dimensions.
    """
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns, dtype=jnp.float32)

    def coulomb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, jnp.inf)

    def compute_batch(start_idx, end_idx):
        # Compute the cost matrix for the batch
        cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulomb_pairwise(x_, y_))(x[start_idx:end_idx]))(x[start_idx:end_idx])
        return cost_m

    for i in range(0, n, batch_size):
        start_idx = i
        end_idx = min(i + batch_size, n)

        cost_m_batch = compute_batch(start_idx, end_idx)

        for j in range(N):
            for k in range(j + 1, N):
                axis = list(range(j)) + list(range(j+1, k)) + list(range(k + 1, N))
                expanded_shape = list(total_cost.shape)
                expanded_shape[j] = end_idx - start_idx
                expanded_shape[k] = end_idx - start_idx
                expanded_shape += [1] * (N - 2)  # Add singleton dimensions for remaining axes

                # Expand dimensions and add the batch cost matrix to the total cost
                cost_m_expanded = jnp.expand_dims(cost_m_batch, axis=axis)
                total_cost = total_cost.at[tuple(slice(None) for _ in range(N))].add(cost_m_expanded)

    return total_cost

@jax.jit
def remove_tensor_sum(c, potentials):
    k = len(potentials)
    for i in range(k):
        c -= jnp.expand_dims(potentials[i], axis=list(range(i)) + list(range(i + 1, k)))
    return c

def tensor_marginal(coupling: jnp.ndarray, slice_index: int) -> jnp.ndarray:
    k = coupling.ndim
    axis = list(range(slice_index)) + list(range(slice_index + 1, k))
    return coupling.sum(axis=axis)

@jax.jit
def tensor_marginals(tensor: jnp.ndarray) -> tuple:
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

def coupling_tensor(potentials, cost_t, epsilon: jnp.float32) -> jnp.ndarray:
    return jnp.exp(-remove_tensor_sum(cost_t, potentials) / epsilon)

def sinkhorn_logsumexp(marginals, c, reg, precision=1e-3, max_iters=350, batch_size=10):
    precision = jnp.float32(precision)
    reg = jnp.float32(reg)
    logging.info("Starting Sinkhorn algorithm")
    start_time = time.time()
    start_mem = get_memory_usage()
    
    N = len(marginals)
    n = marginals[0].shape[0]

    potentials = [jnp.zeros(n, jnp.float32) for _ in range(N)]
    logging.info(f"Initialized potentials with shape {n} for {N} marginals")

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
            potentials_minus_c = remove_tensor_sum(c, potentials)
            lse = logsumexp(potentials_minus_c / -reg, axis=axis)
            potentials[i] += reg * jnp.log(marginals[i]) - reg * lse
        
        return potentials, iter, errors

    # Processing in batches
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_marginals = [m[start:end] for m in marginals]
        batch_c = c[start:end]

        potentials, iter, errors = jax.lax.while_loop(
            lambda var: jnp.logical_and(jnp.max(var[2]) > precision, var[1] <= max_iters),
            lambda var: body_fn(var),
            (potentials, 0, jnp.full(N, jnp.inf))
        )

    P = coupling_tensor(potentials, c, reg)
    end_mem = get_memory_usage()

    log_data = {
        'steps': iter,
        'time': time.time() - start_time,
        'errors': errors,
        'memory_start': start_mem,
        'memory_end': end_mem,
        'memory_usage': end_mem - start_mem,
    }
    
    logging.info(f"Completed Sinkhorn algorithm in {log_data['steps']} steps and {log_data['time']:.2f} seconds")
    logging.info(f"Memory usage: Start: {log_data['memory_start']:.2f} MB, End: {log_data['memory_end']:.2f} MB, Delta: {log_data['memory_usage']:.2f} MB")
    logging.info(f"Final errors: {log_data['errors']}")

    return P, log_data

# Example usage:
x = jnp.linspace(-5, 5, 100)
mu_1 = norm.pdf(x, loc=1.5, scale=0.5)
mu_2 = norm.pdf(x, loc=2.5, scale=0.9)
mu_3 = norm.pdf(x, loc=2, scale=0.6)

mu_1 = mu_1 / mu_1.sum()
mu_2 = mu_2 / mu_2.sum()
mu_3 = mu_3 / mu_3.sum()

marginals = [mu_1, mu_2, mu_3]

batch_size = 20  # Adjust batch size as needed
coulomb_cost = compute_cost_matrix_coulomb_jax(x, len(marginals), batch_size)
P, log_data = sinkhorn_logsumexp(marginals, coulomb_cost, reg=1, precision=1e-5, max_iters=1000, batch_size=batch_size)
