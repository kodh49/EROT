# import all necessary external dependencies
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from env.lib import *

warnings.filterwarnings("ignore")

# configure Loguru logger
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO"
)

# compute partial trace of the given tensor with respect to the first Hilbert space X


# Run Cyclic Projection with cost tensor C and two density matrices 
def quadratic_cyclic_projection(C:jnp.ndarray, marg:list, epsilon:jnp.float64, num_iter:jnp.int16,
                                    convergence_error:jnp.float64):
    # initialize variables
    a, b = marg[0], marg[1] # density matrices
    n, m = a.size(0), b.size(0) # dimension of each density matrix
    U, V = jnp.zeros_like(a, dtype=jnp.complex128), jnp.zeros_like(b, dtype=jnp.complex128) # potentials
    gamma = jnp.zeros_like(C, dtype=jnp.complex128)
    for _ in range(num_iter):
        # calculate normalizing constant lambda
        lmbda = (jnp.trace(jnp.kron(u,v)-C) - epsilon)/(n*m)
        # update lagrangian multipliers u and v
        v = (epsilon * b - jnp.einsum(jnp.kron(u,jnp.eye(n)) - C - lmbda*jnp.eye(n*m), n))/n
        u = (epsilon * a - jnp.einsum(jnp.kron(v,jnp.eye(m)) - C - lmbda*jnp.eye(n*m), m))/m
        # update the shape of optimizer
        gamma = (jnp.kron(u,jnp.eye(n)) + jnp.kron(v,jnp.eye(m)) - C - lmbda * jnp.eye(n*m)).clamp(min=0)/epsilon
        # check for convergence
        if compute_error(gamma, a, b) < convergence_error:
            break
    logger.success(f"Computed Cyclic Projection with error {compute_error(gamma=gamma, rho_1=a, rho_2=b)}.")
    cyclic_projection = gamma.cpu()
    jnp.cuda.empty_cache()
    return cyclic_projection

def compute_error(gamma: jnp.Tensor, rho_1: jnp.Tensor, rho_2: jnp.Tensor) -> jnp.float64:
    """
    Computes L1 norm of the difference between partial traces and marginal distributions
    """
    n, m = rho_1.size(0), rho_2.size(0)
    return 

# Run Gradient Descent
def quadratic_gradient_descent(C:jnp.Tensor, marg:list, epsilon:jnp.float32, gpu:int, num_iter:int,
                                    convergence_error:float) -> jnp.Tensor:
    rho_1, rho_2 = marg[0], marg[1]
    n, m = rho_1.size(0), rho_2.size(0)
    # Transfer to GPU if applicable
    device = jnp.device(f'cuda:{gpu}' if jnp.cuda.is_available() else 'cpu')
    C, rho_1, rho_2, u, v = map(lambda x: x.to(device), [C, rho_1, rho_2, jnp.zeros_like(rho_1), jnp.zeros_like(rho_2)])
    for _ in range(num_iter):
        # add something here. What is a gradient of the objective function?
        a=10

# Run Nesterov Gradient Descent