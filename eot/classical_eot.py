# import all necessary external dependencies
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from env.lib import *

# import internal dependencies
import utils

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

def shannon_sinkhorn(marginals, C, epsilon:float, convergence_error:float, max_iters: int):
    start_time = time.time()
    N = len(marginals)
    n = marginals[0].shape[0]
    vars = [jnp.ones(n) for _ in range(N)]
    K = jnp.exp(-C/epsilon) # Kernel tensor
    error = convergence_error*2
    iterations = 0
    for t in range(1, max_iters+1):
        for i in range(N):
            args = utils.construct_arguments(vars, i)
            vars[i] = marginals[i] / (jnp.einsum(K, jnp.arange(N), *args))
        error = sinkhorn_compute_error(vars, marginals, K)
        if error <= convergence_error:
            iterations = t
            break
    P = compute_P(vars, K)
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Sinkhorn | Elapsed: {time_taken} | Precision: {error}.")
    return P, error, iterations, time_taken

def quadratic_cyclic_projection(C, marg, epsilon: float, num_iter: int = 50000,
                                    convergence_error: float = 1e-9):
    """
    Run Cyclic Projection Algorithm
    """
    start_time = time.time() # measure performance

    # intialize variables
    a, b = marg[0], marg[1] # marginal distributions
    n, m = a.shape[0], b.shape[0] # dimensions of marginals
    error, iterations = convergence_error * 2, 0 # error and number of iterations
    f, g = jnp.zeros_like(a), jnp.zeros_like(b) # dual functionals
    P = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon # coupling
    P_0 = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon

    @jax.jit
    def _quadratic_cyclic_projection(f, g):
        rho = -jnp.clip(-(f[:, None] + g[None, :] - C), a_max=0)
        f = (epsilon * a - jnp.sum(rho + g[None, :] - C, axis=1)) / m
        g = (epsilon * b - jnp.sum(rho + f[:, None] - C, axis=0)) / n
        return f, g

    # projection iterations
    for t in trange(1, num_iter + 1):
        f, g = _quadratic_cyclic_projection(f, g) # update dual functionals
        P = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon # update the coupling
        error = quadratic_compute_error(P, a, b) # update the Frobenius error
        if error < convergence_error:
            iterations = t
            break

    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Cyclic Projection | Elapsed: {time_taken} | Precision: {error}.")

    return P, error, iterations, time_taken

def quadratic_gradient_descent(C: jnp.ndarray, marg: list, epsilon: float, num_iter: int = 50000,
                                   convergence_error: float = 1e-9):
    """
    Run Gradient Descent Algorithm
    """
    start_time = time.time() # measure performance

    # initialize variables
    a, b = marg[0], marg[1] # marginal distributions
    n, m = a.shape[0], b.shape[0] # dimensions of marginals
    step = 1.0 / (m + n) # gradient descent step size
    error, iterations = convergence_error * 2, 0 # error and number of iterations
    f, g = jnp.zeros_like(a), jnp.zeros_like(b) # dual functionals
    P = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon # coupling

    
    @jax.jit # single gradient descent update
    def _quadratic_gradient_descent(f, g):
        f = f - step * epsilon * (jnp.sum(P, axis=1) - a)
        g = g - step * epsilon * (jnp.sum(P, axis=0) - b)
        return f, g

    # gradient descent iterations
    for t in trange(num_iter):
        f, g = _quadratic_gradient_descent(f, g) # update dual functionals
        P = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon # update the coupling
        error = quadratic_compute_error(P, a, b) # update Frobenius error
        if error < convergence_error:
            iterations = t
            break

    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Gradient Descent | Elapsed: {time_taken} | Precision: {error}.")
    return P, error, iterations, time_taken

@jax.jit
def quadratic_fixed_point_iteration(C, marg, epsilon, num_iter, convergence_error):
    
    start_time = time.time() # measure performance

    # initialize variables
    a, b = marg[0], marg[1] # marginal distributions
    n, m = a.shape[0], b.shape[0] # dimensions of marginals
    f, g = jnp.zeros_like(a), jnp.zeros_like(b) # dual functionals
    error, iterations = convergence_error*2, 0 # error and number of iterations
    P = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon # coupling

    @jax.jit # single fixed point iteration
    def _quadratic_fixed_point_iteration(f, g):
        v = -epsilon * (jnp.sum(P, axis=1) - a)
        f += (v - jnp.mean(v)) / m
        u = -epsilon * (jnp.sum(P, axis=0) - b)
        g += (u - jnp.mean(u)) / n
        return f, g
    
    # Run over fixed point iterations
    for t in trange(num_iter):
        f, g = _quadratic_fixed_point_iteration(f, g) # update dual functionals
        P = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon # update the coupling
        error = quadratic_compute_error(P, a, b) # update the Frobenius error
        if error < convergence_error:
            iterations = t
            break

    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Fixed Point Iteration | Elapsed: {time_taken} | Precision: {error}.")

    return P, error, iterations, time_taken

@jax.jit
def quadratic_nesterov_gradient_descent(C, marg: list, epsilon: float, num_iter: int,
                                        convergence_error: float = 1e-9):
    """
    Runs Nesterov Gradient Descent Algorithm
    Input:
    Output:
    """
    start_time = time.time() # measure performance

    # initialize variables
    a, b = marg[0], marg[1] # marginal distributions
    n, m = a.shape[0], b.shape[0] # dimensions of marginals
    step = 1.0 / (m + n) # nesterov step size
    error, iterations = convergence_error * 2, 0 # error and number of iterations    
    f, f_previous, g, g_previous = jnp.zeros_like(a), jnp.zeros_like(a), jnp.zeros_like(b), jnp.zeros_like(b) # dual functionals
    P = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon # coupling

    @jax.jit # single iteration of nesterov gradient descent
    def _quadratic_nesterov_gradient_descent(f, g, f_previous, g_previous):
        # Compute the Nesterov updates
        f_p = f + n * (f - f_previous) / (n + 3)
        g_p = g + n * (g - g_previous) / (n + 3)
        # update dual functionals
        f_new = f_p - step * epsilon * (jnp.sum(P, axis=1) - a)
        g_new = g_p - step * epsilon * (jnp.sum(P, axis=0) - b)
        return f_new, g_new, f, g

    for t in trange(num_iter):
        f, g, f_previous, g_previous = _quadratic_nesterov_gradient_descent(f, g, f_previous, g_previous)
        P = jnp.clip((f[:, None] + g[None, :] - C), a_min=0) / epsilon # update the coupling
        error = quadratic_compute_error(P, a, b) # update the Frobenius error
        if error < convergence_error:
            iterations = t
            break

    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Nesterov Gradient Descent | elapsed time: {time_taken} | error: {error}.")

    return P, error, iterations, time_taken

@jax.jit
def quadratic_compute_error(P: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.float64:
    """
    Computes maximum of two Frobenius norm as max(||P-a||, ||P-b||)
    """
    error_a = jnp.abs(P.sum(axis=1) - a).sum()
    error_b = jnp.abs(P.sum(axis=0) - b).sum()
    return jnp.maximum(error_a, error_b)

@jax.jit
def sinkhorn_compute_error(vars: list, marginals: list, K: int) -> jnp.float64:
    """
    Computes multi-marginal Frobenius norm
    """
    error_value = 0.0
    N = len(marginals)
    P = jnp.einsum(K, np.arange(N), *utils.get_all_arguments(vars), np.arange(N))
    for i in range(N):
      new_error = jnp.sum(jnp.abs(jnp.einsum(P, np.arange(N), [i]) - marginals[i]))
      error_value = jax.lax.select(error_value < new_error, new_error, error_value)
    return error_value

@jax.jit
def compute_P(vars: list, K: int) -> jnp.ndarray:
    """
    Computes a coupling tensor P for the Sinkhorn Algorithm
    """
    N = len(vars)
    return jnp.einsum(K, np.arange(N), *utils.get_all_arguments(vars), np.arange(N))