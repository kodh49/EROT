# Toy samples for EROT demonstration
This folder contains multiple sample files for testing EROT implementations under various scenarios.

## Classical Entropy Regularized Optimal Transport
The cost matrix $C \in \mathbb{R}^{100\times100}$, stored as a pytorch tensor file `C.pt`, is a $100 \times 100$ sized real matrix containing Euclidean distances between each pair of indices, i.e. $\forall i,j \in \{1, \cdots, 100\}$,
$$
C_{ij} = \lvert i-j \rvert^2
$$
Two probability vectors $X, Y \in \mathbb{R}^{100}$, stored as two pytorch tensor files `X.pt` and `Y.pt`, follow two distributions
$$
X \sim \mathcal{N}(\mu_X=2.1,\sigma_X=1)
\hspace{50pt}
Y \sim \mathcal{N}(\mu_Y=-3, \sigma_Y=0.75)
$$
where $\mathcal{N}(\mu=0,\sigma=1)$ denotes a normal distribution with mean of 0 and standard deviation of 1.

## Quantum Entropy Regularized Optimal Transport
To be implemented.