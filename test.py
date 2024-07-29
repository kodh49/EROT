import torch
import os

# Set the environment variable before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"

# Set environment variables to use all 128 CPU cores
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=128'
os.environ['OMP_NUM_THREADS'] = '128'

def print_tensor(loc: str) -> None:
    T = torch.load(loc)
    print(T)

import jax
import jax.numpy as jnp

# Function to perform some computation
def compute(x):
    return jnp.sin(x) * jnp.cos(x)

def repeat(x):
    for _ in range(300_000_000):
        for _ in range(300_300_300):
            for _ in range(300_300_300):
                compute(x)

# Generate random data
key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (128, 100000))  # Adjust size as needed

# Parallelize the computation across multiple devices (CPU cores)
result = jax.pmap(compute)(data)

print(result)