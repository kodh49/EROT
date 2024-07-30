import os

# Set the environment variable before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"


import jax
import jax.numpy as jnp

def print_tensor(loc: str) -> None:
    T = jnp.load(loc)
    print(T)

print_tensor("strong.npy")