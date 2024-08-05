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

def measure_performance(cost_matrix: jnp.ndarray, algorithm, algorithm_args: list, epsilon, precision_values=[1e-3, 1e-4, 1e-5]):
    results = []

    for precision in precision_values:
        P, error, iterations, time_taken = algorithm(algorithm_args, cost_matrix, epsilon, precision)

        results.append({
            'Regularization': epsilon,
            'Precision': precision,
            'Error': error,
            'Iterations': iterations,
            'Time_taken': time_taken
        })

    df_results = pd.DataFrame(results)
    return df_results

# Example usage:

# Compute results for coulomb_cost
coulomb_cost = compute_cost_matrix_coulomb(100, 2)
df_results_coulomb = measure_performance(coulomb_cost)

# Compute results for quadratic_cost
quadratic_cost = compute_cost_matrix_quadratic(100, 2)
df_results_quadratic = measure_performance(quadratic_cost)

# Merge or concatenate the two DataFrames
df_2m = pd.concat([df_results_coulomb, df_results_quadratic], keys=['Coulomb 2 marginals', 'Quadratic 2 marginals'])

df_2m