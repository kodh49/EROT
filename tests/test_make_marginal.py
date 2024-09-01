import unittest
import os
import itertools
import jax.numpy as jnp
import shutil
from unittest.mock import patch, MagicMock
from subprocess import run, CalledProcessError

# Test Class for make_marginal_tensor
class TestMakeMarginalTensor(unittest.TestCase):
    
    def setUp(self):
        # Set up paths and parameters
        self.cpath = os.path.dirname(os.path.realpath(__file__))
        self.project_path = os.path.join(self.cpath, '..')
        
        # Regularization parameters
        self.epsilon_list = [1, 1e-2, 1e-4]
        
        # Marginal distribution paths
        self.mu_list = [
            os.path.join(self.cpath, 'unittests_data/mu_1.npy'),
            os.path.join(self.cpath, 'unittests_data/mu_2.npy'),
            os.path.join(self.cpath, 'unittests_data/mu_3.npy')
        ]
        
        # Cost tensor paths for different numbers of marginals
        self.cost_paths_N2 = {
            "quadratic": os.path.join(self.cpath, 'unittests_data/quadratic_N2.npy'),
            "strong_coulomb": os.path.join(self.cpath, 'unittests_data/strong_coulomb_N2.npy'),
            "weak_coulomb": os.path.join(self.cpath, 'unittests_data/weak_coulomb_N2.npy')
        }
        self.cost_paths_N3 = {
            "quadratic": os.path.join(self.cpath, 'unittests_data/quadratic_N3.npy'),
            "strong_coulomb": os.path.join(self.cpath, 'unittests_data/strong_coulomb_N3.npy'),
            "weak_coulomb": os.path.join(self.cpath, 'unittests_data/weak_coulomb_N3.npy')
        }
        self.cost_paths_N4 = {
            "quadratic": os.path.join(self.cpath, 'unittests_data/quadratic_N4.npy'),
            "strong_coulomb": os.path.join(self.cpath, 'unittests_data/strong_coulomb_N4.npy'),
            "weak_coulomb": os.path.join(self.cpath, 'unittests_data/weak_coulomb_N4.npy')
        }
        
        # Output directory
        self.output_dir = os.path.join(self.cpath, 'output_results')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def run_test(self, alg, mu_combination, epsilon, cost_name, cost_path, n_marginals):
        try:
            # Construct the output file path
            output_file = os.path.join(
                self.output_dir,
                f'result_{alg}_{cost_name}_eps{epsilon}_mu{"_".join([os.path.basename(mu) for mu in mu_combination])}.npy'
            )
            # Construct the command
            command = f"python ./eot/run_classical_eot.py --alg {alg} --cost {cost_path} --marginal {' '.join(mu_combination)} --epsilon {epsilon} --out {output_file}"
            run(command, shell=True, check=True)
            
            # Load reference and result tensors
            reference = jnp.load(f"test_{output_file}")  # Load reference tensor
            result = jnp.load(output_file)  # Load test result tensor
            
            # Assert that they are close enough
            self.assertTrue(jnp.allclose(reference, result, atol=1e-05), "Results do not match reference.")
        
        except CalledProcessError as e:
            self.fail(f"Command failed: {e}")

    def test_N2(self):
        for mu_combination in itertools.combinations(self.mu_list, 2):
            for epsilon in self.epsilon_list:
                for cost_name, cost_path in self.cost_paths_N2.items():
                    self.run_test('sinkhorn', mu_combination, epsilon, cost_name, cost_path, 2)

    def test_N3(self):
        for mu_combination in itertools.combinations(self.mu_list, 3):
            for epsilon in self.epsilon_list:
                for cost_name, cost_path in self.cost_paths_N3.items():
                    self.run_test('sinkhorn', mu_combination, epsilon, cost_name, cost_path, 3)

    def test_N4(self):
        for mu_combination in itertools.combinations(self.mu_list, 4):
            for epsilon in self.epsilon_list:
                for cost_name, cost_path in self.cost_paths_N4.items():
                    self.run_test('sinkhorn', mu_combination, epsilon, cost_name, cost_path, 4)

    def test_additional_algorithms(self):
        algorithms = ['fpi', 'cyc-proj', 'gradient-descent', 'nesterov']
        for alg in algorithms:
            self.test_N2()
            # Add tests for N3 and N4 if needed for these algorithms


if __name__ == "__main__":
    unittest.main()
