import utils
from ..env.dependencies import *

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

@jax.jit
def quadratic_compute_error(P: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.float64:
    """
    Computes maximum of two Frobenius norm as max(||P-a||, ||P-b||)
    """
    error_a = jnp.abs(P.sum(axis=1) - a).sum()
    error_b = jnp.abs(P.sum(axis=0) - b).sum()
    return jnp.maximum(error_a, error_b)

# Run Cyclic Projection with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_cyclic_projection(C:torch.Tensor, marg:list, epsilon:torch.float32, gpu:int, num_iter:int,
                                         convergence_error:float) -> torch.Tensor:
    start_time = time.time()
    a, b = marg[0], marg[1]
    n, m = a.size(0), b.size(0)
    error = convergence_error*2
    iterations = 0
    for t in range(1, num_iter+1):
        # Calculate rho and update f and g
        rho = -(f[:, None] + g[None, :] - C).clamp(max=0)
        f = (epsilon * a - (rho + g[None, :] - C).sum(dim=1)) / m
        g = (epsilon * b - (rho + f[:, None] - C).sum(dim=0)) / n
        # Check for convergence based on L^1 norm of projections
        P = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon)
        error = quadratic_compute_error(P, a, b)
        if error < convergence_error:
            iterations = t
            break
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Cyclic Projection | Elapsed: {time_taken} | Precision: {error}.")
    return P, error, iterations, time_taken

# Run Gradient Descent with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_gradient_descent(C: torch.Tensor, marg:list, epsilon:torch.float32, gpu:int, num_iter:int,
                                        convergence_error:float) -> torch.Tensor:
    start_time = time.time()
    a, b = marg[0], marg[1]
    n, m =a.size(0), b.size(0)
    step = 1.0 / (m + n)
    error = convergence_error*2
    iterations = 0
    for t in range(num_iter): 
        # Calculate P and update f and g
        P = (f[:, None] + g[None, :] - C).clamp(min=0) / epsilon
        f -= step * epsilon * (P.sum(dim=1) - a)
        g -= step * epsilon * (P.sum(dim=0) - b)
        # Check for convergence based on L^1 norm of projections
        error = quadratic_compute_error(P, a, b)
        if error < convergence_error:
            iterations = t
            break
    P = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon) # Retrieve primal result from dual maximization
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Gradient Descent | Elapsed: {time_taken} | Precision: {error}.")
    return P, error, iterations, time_taken

# Run Fixed Point Iteration with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_fixed_point_iteration(C:torch.Tensor, marg:list, epsilon: torch.float32, gpu:int, num_iter:int,
                                             convergence_error:float) -> torch.Tensor:
    start_time = time.time()
    a, b = marg[0], marg[1]
    n, m = a.size(0), b.size(0)
    for _ in range(num_iter):
        # Calculate P and update f and g
        P = (f[:, None] + g[None, :] - C).clamp(min=0) / epsilon
        v = -epsilon * (P.sum(dim=1) - a)
        f += (v - v.mean()) / m
        u = -epsilon * (P.sum(dim=0) - b)
        g += (u - u.mean()) / n
        # Check for convergence based on L^1 norm of projections
        if quadratic_compute_error(P, a, b) < convergence_error:
            break
    fixed_point_iteration = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon)  # Retrieve result to CPU
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Fixed Point Iteration | Elapsed: {time_taken} | Precision: {error}.")
    return fixed_point_iteration

# Run Nesterov Gradient Descent with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_nesterov_gradient_descent(C: torch.Tensor, marg: list, epsilon: torch.float32, gpu: int, num_iter: int,
                                                 convergence_error: float) -> torch.Tensor:
    start_time = time.time()
    a, b = marg[0], marg[1]
    n, m =a.size(0), b.size(0)
    step = 1.0 / (m + n)
    f, g = torch.zeros_like(a), torch.zeros_like(b)
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
        if quadratic_compute_error(P, a, b) < convergence_error:
            break
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Nesterov Gradient Descent | elapsed time: {time_taken} | error: {quadratic_compute_error(nesterov_gradient_descent, a,b)}.")
    nesterov_gradient_descent = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon) # Retrieve primal result from dual maximization
    return nesterov_gradient_descent