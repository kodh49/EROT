# Test Data for Classical and Quantum Optimal Transport Algorithms
This folder contains multiple test cost tensor and marginal distributions for the unit test of EROT implementations under various scenarios.

## Classical Entropy Regularized Optimal Transport
The classical side of unit test performs multiple testings of EROT implementations under various cost tensor and marginal probability distributions.

### Cost Tensor
The unit test provides cost matrix $C$ with varying number of datapoints and dimensions.

- Strong Coulomb Cost
- Weak Coulomb Cost
- Euclidean Cost
- Quadratic Cost

### Marginal Probability Distributions
The unit test uses discretized 1 dimensional Gaussian probability distributions as marginals.

- mu_1
- mu_2
- mu_3
- mu_4
- mu_5

### Regularization Parameters
For the sake of complexity, unit test only tests classical optimal transport algorithms with relatively large regularization parameters $\epsilon = [0.001, 0.01, 0.1, 1, 5, 10]$.

## Quantum Entropy Regularized Optimal Transport
The quantum side of unit test performs multiple testings under various Hamiltonian and density matrices.

### Hamiltonian Tensor
The unit test provides Hamiltonian cost tensor $H$ of systems with varying number of particles with a total of 6 particles at maximum.

- Ising Model
- Hubbard Model

### Marginal Density Matrices


### Regularization Parameters
For the sake of complexity, unit test only tests quantum optimal transport algorithms with relatively large regularization parameters $\epsilon = [1, 5, 10]$.