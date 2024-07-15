import torch
import warnings, sys
from loguru import logger

warnings.filterwarnings("ignore")

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

# Run Cyclic Projection with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_cyclic_projection(C:torch.Tensor, marg:list, epsilon:torch.float32, gpu:int, num_iter:int,
                                         convergence_error:float) -> torch.Tensor:
    a, b = marg[0], marg[1]
    n, m = a.size(0), b.size(0)
    # Transfer to GPU if applicable
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    C, a, b, f, g = map(lambda x: x.to(device), [C, a, b, torch.zeros_like(a), torch.zeros_like(b)])
    for _ in range(num_iter):
        # Store previous values in simple tensors
        f_prev, g_prev = f.detach().clone(), g.detach().clone()
        # Calculate rho and update f and g
        rho = -(f[:, None] + g[None, :] - C).clamp(max=0)
        f = (epsilon * a - (rho + g[None, :] - C).sum(dim=1)) / m
        g = (epsilon * b - (rho + f[:, None] - C).sum(dim=0)) / n
        # Check for convergence based on L^1 norm of projections
        P = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon)
        if torch.abs((P.sum(dim=1) - a)).sum() < convergence_error and torch.abs((P.sum(dim=0) - b)).sum() < convergence_error:
            break
    cyclic_projection = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon)
    logger.success(f"Computed Cyclic Projection with error: {compute_error(cyclic_projection, a,b)}.")
    cyclic_projection = cyclic_projection.cpu()
    torch.cuda.empty_cache()
    return cyclic_projection

# Run Gradient Descent with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_gradient_descent(C: torch.Tensor, marg:list, epsilon:torch.float32, gpu:int, num_iter:int,
                                        convergence_error:float) -> torch.Tensor:
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
        if torch.abs((P.sum(dim=1) - a)).sum() < convergence_error and torch.abs((P.sum(dim=0) - b)).sum() < convergence_error:
            break
    gradient_descent = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon) # Retrieve result to CPU
    logger.success(f"Computed Gradient Descent with error: {compute_error(gradient_descent, a,b)}.")
    gradient_descent = gradient_descent.cpu()
    torch.cuda.empty_cache()
    return gradient_descent

# Run Fixed Point Iteration with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_fixed_point_iteration(C:torch.Tensor, marg:list, epsilon: torch.float32, gpu:int, num_iter:int,
                                             convergence_error:float) -> torch.Tensor:
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
        if torch.abs((P.sum(dim=1) - a)).sum() < convergence_error and torch.abs((P.sum(dim=0) - b)).sum() < convergence_error:
            break
    fixed_point_iteration = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon)  # Retrieve result to CPU
    logger.success(f"Computed Fixed Point Iteration with error {compute_error(fixed_point_iteration, a,b)}.")
    fixed_point_iteration = fixed_point_iteration.cpu()
    torch.cuda.empty_cache()
    return fixed_point_iteration

# Run Nesterov Gradient Descent with cost tensor C and 2 marginal vectors marg[0,1] on cuda:gpu
def quadratic_nesterov_gradient_descent(C: torch.Tensor, marg: list, epsilon: torch.float32, gpu: int, num_iter: int,
                                                 convergence_error: float) -> torch.Tensor:
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
        if torch.abs((P.sum(dim=1) - a)).sum() < convergence_error and torch.abs((P.sum(dim=0) - b)).sum() < convergence_error:
            break
    nesterov_gradient_descent = ((f[:, None] + g[None, :] - C).clamp(min=0) / epsilon)
    logger.success(f"Computed Nesterov Gradient Descent with error {compute_error(nesterov_gradient_descent, a, b)}.")
    nesterov_gradient_descent = nesterov_gradient_descent.cpu()
    torch.cuda.empty_cache()
    return nesterov_gradient_descent


def compute_error(P: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.float64:
    return max(torch.abs((P.sum(dim=1) - a)).sum(), torch.abs((P.sum(dim=0) - b)).sum())