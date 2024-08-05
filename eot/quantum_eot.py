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

@jax.jit
def tr_1(A: jnp.ndarray) -> jnp.ndarray:
    """
    computes first partial trace of an operator A
    """
    n = int(jnp.sqrt(jnp.shape(A)[0])) # reshape A into a rank 4 tensor
    return jnp.einsum('ijkl->jl', A.reshape(n,n,n,n)) / n

@jax.jit
def tr_2(A: jnp.ndarray) -> jnp.ndarray:
    """
    computes second partial trace of an operator A
    """
    n = int(jnp.sqrt(jnp.shape(A)[0])) # reshape A into a rank 4 tensor
    return jnp.einsum('ijkl->ik', A.reshape(n,n,n,n)) / n

@jax.jit
def grad_U(Gamma: jnp.ndarray, rho_1: jnp.ndarray, epsilon: float) -> jnp.ndarray:
    """
    computes gradient of the dual functional with respect to U
    """
    return epsilon * rho_1 - tr_2(Gamma)

@jax.jit
def grad_V(Gamma: jnp.ndarray, rho_2: jnp.ndarray, epsilon: float) -> jnp.ndarray:
    """
    computes gradient of the dual functional with respect to V
    """
    return epsilon * rho_2 - tr_1(Gamma)

@jax.jit
def change_of_basis_matrix(P_old: jnp.ndarray, P_new: jnp.ndarray) -> jnp.ndarray:
	"""
	Computes a change of basis matrix to switch from P_old to P_new
	"""
	return jnp.linalg.solve(P_old, P_new)

@jax.jit
def represent(A: jnp.ndarray, change_of_basis_matrix: jnp.ndarray) -> jnp.ndarray:
	"""
	Computes the new representation of an operator A under change_of_basis matrix
	"""
	return jnp.linalg.solve(change_of_basis_matrix, A @ change_of_basis_matrix)

@jax.jit
def represent_H(A: jnp.ndarray, change_of_basis_matrix: jnp.ndarray) -> jnp.ndarray:
	"""
	Computes the new representation of an operator A on the first Hilbert space H
	"""
	n = jnp.shape(A)[0]
	lifted_A = jnp.kron(A, jnp.eye(n))
	return tr_2(np.linalg.solve(change_of_basis_matrix, lifted_A @ change_of_basis_matrix))

@jax.jit
def represent_K(A: jnp.ndarray, change_of_basis_matrix: jnp.ndarray) -> jnp.ndarray:
	"""
	Computes the new representation of an operator A on the second Hilbert space K
	"""
	n = jnp.shape(A)[0]
	lifted_A = jnp.kron(jnp.eye(n), A)
	return tr_1(np.linalg.solve(change_of_basis_matrix, lifted_A @ change_of_basis_matrix))

@jax.jit
def clip_diagonalize(A: jnp.ndarray) -> jnp.ndarray:
    """
    Diagonalize and clip the matrix A to have nonnegative eigenvalues
    """
    return jnp.diag(jnp.clip(jnp.linalg.eigvalsh(A), a_min=0, a_max=None))

@jax.jit
def diagonalize(A: jnp.ndarray):
    """
    Diagonalize the matrix A such that A=PDP^{-1}
    """
    D, P = jnp.linalg.eigh(A)
    return jnp.diag(D), P

@jax.jit
def hilbert_schmidt(A: jnp.ndarray) -> float:
    """
    Compute Hilbert-Schmidt norm of a matrix A
    """
    return jnp.sqrt(jnp.trace(A.conjugate().T @ A).abs())

@jax.jit
def compute_error(Gamma: jnp.ndarray, rho_1: jnp.ndarray, rho_2: jnp.ndarray) -> float:
    """
    Compute the error of the quantum gradient descent
    """
    diff_H = tr_1(Gamma) - rho_2 # error on the first hilbert space H
    diff_K = tr_2(Gamma) - rho_1 # error on the second hilbert space K
    return max(hilbert_schmidt(diff_H), hilbert_schmidt(diff_K))

def quantum_gradient_descent(C: jnp.ndarray, rho_1: jnp.ndarray, rho_2: jnp.ndarray, epsilon: float = 1, 
                             num_iter: int = 50000, convergence_error: float = 1e-9):
    """
	Run Quantum Gradient Descent algorithm for Quadratic Regularization
	"""
    start_time = time.time()
    n, m = rho_1.shape[0], rho_2.shape[0]
    step_size = 1.0 / (m*n)
    error, iterations = convergence_error * 2, 0
    U, V = jnp.zeros_like(rho_1), jnp.zeros_like(rho_2)
    P_old, P_new = jnp.eye(n*m), jnp.eye(n*m)
    Gamma = jnp.zeros_like(C)
    for k in trange(num_iter):
        # update quantum coupling
        Gamma = jnp.kron(U, jnp.eye(m)) + jnp.kron(jnp.eye(n), V) - C
        Gamma = clip_diagonalize(Gamma) # normalize for nonnegative eigenvalues
        Gamma, P_new = diagonalize(Gamma) # obtain eigenbasis for updated Gamma
        # change the representation into a new basis
        Q = change_of_basis_matrix(P_old, P_new) # obtain change of basis matrix
        U = represent_H(U, Q)
        V = represent_K(V, Q)
        rho_1 = represent_H(rho_1, Q)
        rho_2 = represent_K(rho_2, Q)
        # compute error of the updated Gamma and decide termination
        error = compute_error(Gamma, rho_1, rho_2)
        if error < convergence_error:
            iterations = k
            break
        # gradient descent updates
        U += step_size * grad_U(Gamma, rho_1, epsilon)
        V += step_size * grad_V(Gamma, rho_2, epsilon)
        P_old = P_new # update the eigenbasis matrix
    end_time = time.time()
    time_taken = end_time - start_time
    logger.success(f"Gradient Descent | Elapsed: {time_taken} | Precision: {error}.")
    return Gamma, error, iterations, time_taken


# Run Nesterov Gradient Descent