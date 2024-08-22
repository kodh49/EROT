# import all necessary external dependencies
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from env.lib import *
import eot.quantum_eot as quantum
import eot.utils as utils

def print_tensor(loc: str) -> None:
    T = jnp.load(loc)
    print(T)

rho_1 = jnp.array([[0.4, 0.1], [0.1, 0.6]])
rho_2 = jnp.array([[0.4, 0.1], [0.1, 0.6]])
cost = jnp.eye(rho_1.shape[0]**2)

Gamma, error, err_lst, iters, elapsed_time = quantum.quantum_gradient_descent(cost, rho_1, rho_2, convergence_error=1e-3, num_iter=500)

x = np.linspace(start=1,stop=len(err_lst), num=len(err_lst), endpoint=True)
plt.plot(x, err_lst, label=rf'Error over iteration')
plt.legend()
plt.show()

print(Gamma)