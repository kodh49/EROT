# import all necessary external dependencies
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from env.lib import *
import eot.quantum_eot as quantum

def print_tensor(loc: str) -> None:
    T = jnp.load(loc)
    print(T)


C = jnp.asarray([
				[  141 , 2+1j ,  3j   ,  4-1j ],
				[ 2-1j ,  56  ,  6-2j ,   7   ],
				[ -3j  , 6+2j ,   80  , 19+1j ],
				[ 4+1j ,  7   , 19-1j ,   16  ]
				])

rho_1 = jnp.asarray([
					[0.3,  0.01+0.02j],
					[0.01-0.02j,  0.7]
					])

rho_2 = jnp.asarray([
					[0.5,  -0.04-0.01j],
					[-0.04+0.01j,  0.5]
					])

Gamma, error, iters, elapsed_time = quantum.quantum_gradient_descent(C, rho_1, rho_2, convergence_error=1e-2, num_iter=500)
print(Gamma)