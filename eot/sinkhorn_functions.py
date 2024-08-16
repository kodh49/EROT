import jax.numpy as jnp
from functools import partial
import jax
from jax.scipy.special import logsumexp

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_coulomb_jax(x, N):
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns)

    def coulomb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, jnp.inf)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulomb_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)

    return total_cost

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_quadratic_jax(x, N):
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns)

    def quadratic_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, diff**2, 0)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: quadratic_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)

    return total_cost

@jax.jit
def remove_tensor_sum(c, u):
    k = len(u)
    for i in range(k):
        c -= jnp.expand_dims(u[i], axis=list(range(i)) + list(range(i + 1, k)))
    return c

@jax.jit
def coupling_tensor(potentials, cost_t, epsilon):
    return jnp.exp(-remove_tensor_sum(cost_t, potentials) / epsilon)

def tensor_marginal(coupling, slice_index):
    k = coupling.ndim
    axis = list(range(slice_index)) + list(range(slice_index + 1, k))
    return coupling.sum(axis=axis)

@jax.jit
def tensor_marginals(tensor):
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
