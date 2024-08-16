import os
import multiprocessing as mp
import numpy as np
import itertools
# Set the environment variable before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"


import jax
import jax.numpy as jnp

def print_tensor(loc: str) -> None:
    T = jnp.load(loc)
    print(T)

def sample(x,y,z):
    xl = jnp.linspace(0, 100, x)
    yl = jnp.linspace(0, 100, y)
    zl = jnp.linspace(0, 100, z)
    for a,b,c in itertools.product(xl,yl,zl):
        w = a * b * c
    return w

if __name__ == '__main__':
    mp.set_start_method('spawn') # prevent collision with jax
    pool_size = mp.cpu_count() # utilize half of the available cpus
    pool = mp.Pool(pool_size // 2)
    param = list(zip(range(100_000), range(100_000), range(100_000)))
    print(pool_size)
    xs = np.array(pool.starmap(sample, param))
