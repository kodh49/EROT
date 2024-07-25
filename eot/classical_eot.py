import torch
import jax
import time
import utils
import numpy as np
import jax.numpy as jnp
import warnings, sys
from loguru import logger

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

# Run Sinkhorn Algorithm with cost tensor C and multi marginal vectors
def shannon_sinkhorn(marginals: list, C, epsilon:float, precision:float, max_iters: int):
    start_time = time.time()
    N = len(marginals)
    n = marginals[0].shape[0]
    vars = [jnp.ones(n) for _ in range(N)]
    K = jnp.exp(-C/epsilon) # Kernel tensor
    error = sinkhorn_compute_error(vars, marginals, K)
    iterations = 0
    for t in range(1, max_iters):
        for i in range(N):
            args = utils.construct_arguments(vars, i)
            vars[i] = marginals[i] / (jnp.einsum(K, np.arange(N), *args))
        # if t % 10 == 0:
        error = sinkhorn_compute_error(vars, marginals, K)
        iterations = t
        if error <= precision:
            break
    P = compute_P(vars, K)
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Sinkhorn | elapsed time: {time_taken} | error: {error}.")
    return P, error, iterations, time_taken

# Run Cyclic Projection with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_cyclic_projection(C:torch.Tensor, marg:list, epsilon:torch.float32, gpu:int, num_iter:int,
                                         convergence_error:float) -> torch.Tensor:
    start_time = time.time()
    a, b = marg[0], marg[1]
    n, m = a.size(0), b.size(0)
    # Transfer to GPU if applicable
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    C, a, b, f, g = map(lambda x: x.to(device), [C, a, b, torch.zeros_like(a), torch.zeros_like(b)])
    for _ in range(num_iter):
        # Calculate rho and update f and g
        rho = -(f[:, None] + g[None, :] - C).clamp(max=0)
        f = (epsilon * a - (rho + g[None, :] - C).sum(dim=1)) / m
        g = (epsilon * b - (rho + f[:, None] - C).sum(dim=0)) / n
        # Check for convergence based on L^1 norm of projections
        P = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon)
        if compute_error(P, a, b) < convergence_error:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Cyclic Projection | elapsed time: {time_taken} | error: {compute_error(P, a,b)}.")
    cyclic_projection = P.cpu()
    torch.cuda.empty_cache()
    return cyclic_projection

# Run Gradient Descent with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_gradient_descent(C: torch.Tensor, marg:list, epsilon:torch.float32, gpu:int, num_iter:int,
                                        convergence_error:float) -> torch.Tensor:
    start_time = time.time()
    a, b = marg[0], marg[1]
    n, m =a.size(0), b.size(0)
    step = 1.0 / (m + n)
    # Transfer to GPU if applicable
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    C, a, b, f, g = map(lambda x: x.to(device), [C, a, b, torch.zeros_like(a), torch.zeros_like(b)])
    for _ in range(num_iter):
        f_prev, g_prev = f.clone(), g.clone()   
        # Calculate P and update f and g
        P = (f[:, None] + g[None, :] - C).clamp(min=0) / epsilon
        f -= step * epsilon * (P.sum(dim=1) - a)
        g -= step * epsilon * (P.sum(dim=0) - b)
        # Check for convergence based on L^1 norm of projections
        if compute_error(P, a, b) < convergence_error:
            break
    gradient_descent = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon) # Retrieve primal result from dual maximization
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Gradient Descent | elapsed time: {time_taken} | error: {compute_error(gradient_descent, a,b)}.")
    gradient_descent = gradient_descent.cpu()
    torch.cuda.empty_cache()
    return gradient_descent

# Run Fixed Point Iteration with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_fixed_point_iteration(C:torch.Tensor, marg:list, epsilon: torch.float32, gpu:int, num_iter:int,
                                             convergence_error:float) -> torch.Tensor:
    start_time = time.time()
    a, b = marg[0], marg[1]
    n, m = a.size(0), b.size(0)
    # Transfer to GPU if applicable
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    C, a, b, f, g = map(lambda x: x.to(device), [C, a, b, torch.zeros_like(a), torch.zeros_like(b)])
    for _ in range(num_iter):
        f_prev, g_prev = f.clone(), g.clone()
        # Calculate P and update f and g
        P = (f[:, None] + g[None, :] - C).clamp(min=0) / epsilon
        v = -epsilon * (P.sum(dim=1) - a)
        f += (v - v.mean()) / m
        u = -epsilon * (P.sum(dim=0) - b)
        g += (u - u.mean()) / n
        # Check for convergence based on L^1 norm of projections
        if compute_error(P, a, b) < convergence_error:
            break
    fixed_point_iteration = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon)  # Retrieve result to CPU
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Fixed Point Iteration | elapsed time: {time_taken} | error: {compute_error(fixed_point_iteration, a,b)}.")
    fixed_point_iteration = fixed_point_iteration.cpu()
    torch.cuda.empty_cache()
    return fixed_point_iteration

# Run Nesterov Gradient Descent with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_nesterov_gradient_descent(C: torch.Tensor, marg: list, epsilon: torch.float32, gpu: int, num_iter: int,
                                                 convergence_error: float) -> torch.Tensor:
    start_time = time.time()
    a, b = marg[0], marg[1]
    n, m =a.size(0), b.size(0)
    step = 1.0 / (m + n)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    C, a, b, f, g = map(lambda x: x.to(device), [C, a, b, torch.zeros_like(a), torch.zeros_like(b)])
    f_previous, g_previous = f.clone(), g.clone()
    for _ in range(num_iter):
        # Compute the Nesterov updates
        f_p = f + n * (f - f_previous) / (n + 3)
        g_p = g + n * (g - g_previous) / (n + 3)
        # Calculate P and update f and g
        P = (f_p[:, None] + g_p[None, :] - C).clamp(min=0) / epsilon
        f_new = f_p - step * epsilon * (P.sum(dim=1) - a)
        g_new = g_p - step * epsilon * (P.sum(dim=0) - b)
        f_previous.copy_(f)
        g_previous.copy_(g)
        f.copy_(f_new)
        g.copy_(g_new)
        # Check for convergence based on L^1 norm of projections
        if compute_error(P, a, b) < convergence_error:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Nesterov Gradient Descent | elapsed time: {time_taken} | error: {compute_error(nesterov_gradient_descent, a,b)}.")
    nesterov_gradient_descent = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon) # Retrieve primal result from dual maximization
    nesterov_gradient_descent = nesterov_gradient_descent.cpu()
    torch.cuda.empty_cache()
    return nesterov_gradient_descent


def compute_error(P: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.float64:
    return max(torch.abs((P.sum(dim=1) - a)).sum(), torch.abs((P.sum(dim=0) - b)).sum())

@jax.jit
def sinkhorn_compute_error(vars, marginals, K):
    error_value = 0.0
    N = len(marginals)
    P = jnp.einsum(K, np.arange(N), *utils.get_all_arguments(vars), np.arange(N))
    for i in range(N):
      new_error = jnp.sum(jnp.abs(jnp.einsum(P, np.arange(N), [i]) - marginals[i]))
      error_value = jax.lax.select(error_value < new_error, new_error, error_value)
    return error_value

@jax.jit
def compute_P(vars, K):
    N = len(vars)
    return jnp.einsum(K, np.arange(N), *utils.get_all_arguments(vars), np.arange(N))
