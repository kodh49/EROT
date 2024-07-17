import torch
import warnings, sys
from loguru import logger

warnings.filterwarnings("ignore")

# configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

# reshape the matrix on the product space into a block form
def block_decomp(matrix: torch.Tensor, n: int):
    return matrix.view(n, n, n, n).permute(0, 2, 1, 3).contiguous()

# compute partial trace of the given tensor with respect to the first Hilbert space X
def tr_X(operator: torch.Tensor, n: int, batch_size: int = 2) -> torch.Tensor:
    """
    Compute partial trace of the given operator represented in n^2 × n^2 matrix with respect to the first Hilbert space X
    """
    block_view = block_decomp(operator, n)
    trace_X = torch.zeros((n, n), device=operator.device)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        trace_X[i:end, :] = block_view[i:end, :, :, :].diagonal(dim1=0, dim2=1).sum(dim=-1)
    return trace_X

def tr_Y(operator: torch.Tensor, n: int, batch_size: int = 2) -> torch.Tensor:
    """
    Compute partial trace of the given operator represented in n^2 × n^2 matrix with respect to the second Hilbert space Y
    """
    block_view = block_decomp(operator, n)
    trace_Y = torch.zeros((n, n), device=operator.device)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        diag_vectors = block_view[i:end, :, :, :].diagonal(dim1=2, dim2=3)
        trace_Y[i:end, :] = diag_vectors.sum(dim=-1)
    return trace_Y

# Run Cyclic Projection with cost tensor C and 2 density matrices
def quadratic_cyclic_projection(C:torch.Tensor, marg:list, epsilon:torch.float32, gpu:int, num_iter:int,
                                    convergence_error:float) -> torch.Tensor:
    a, b = marg[0], marg[1]
    # need another function to check the output
    n, m = a.size(0), b.size(0)
    # Transfer to GPU if applicable
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    C, a, b, u, v = map(lambda x: x.to(device), [C, a, b, torch.zeros_like(a), torch.zeros_like(b)])
    gamma = torch.zeros_like(C, device=device)
    for _ in range(num_iter):
        # calculate normalizing constant lambda
        lmbda = (torch.trace(torch.kron(u,v)-C) - epsilon)/(n*m)
        # update lagrangian multipliers u and v
        v = (epsilon * b - tr_X(torch.kron(u,torch.eye(n)) - C - lmbda*torch.eye(n*m), n))/n
        u = (epsilon * a - tr_Y(torch.kron(v,torch.eye(m)) - C - lmbda*torch.eye(n*m), m))/m
        # update the shape of optimizer
        gamma = (torch.kron(u,torch.eye(n)) + torch.kron(v,torch.eye(m)) - C - lmbda * torch.eye(n*m)).clamp(min=0)/epsilon
        # check for convergence
        if compute_error(gamma, a, b) < convergence_error:
            break
    logger.success(f"Computed Cyclic Projection with error {compute_error(gamma=gamma, rho_1=a, rho_2=b)}.")
    cyclic_projection = gamma.cpu()
    torch.cuda.empty_cache()
    return cyclic_projection

def compute_error(gamma: torch.Tensor, rho_1: torch.Tensor, rho_2: torch.Tensor) -> torch.float64:
    """
    Computes L1 norm of the difference between partial traces and marginal distributions
    """
    n, m = rho_1.size(0), rho_2.size(0)
    return max(torch.abs(tr_Y(gamma, n*m)-rho_1).sum(), torch.abs(tr_X(gamma, n*m)-rho_2).sum())

# Run Gradient Descent
def quadratic_gradient_descent(C:torch.Tensor, marg:list, epsilon:torch.float32, gpu:int, num_iter:int,
                                    convergence_error:float) -> torch.Tensor:
    rho_1, rho_2 = marg[0], marg[1]
    n, m = rho_1.size(0), rho_2.size(0)
    # Transfer to GPU if applicable
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    C, rho_1, rho_2, u, v = map(lambda x: x.to(device), [C, rho_1, rho_2, torch.zeros_like(rho_1), torch.zeros_like(rho_2)])
    for _ in range(num_iter):
        # add something here. What is a gradient of the objective function?
        a=10

# Run Nesterov Gradient Descent