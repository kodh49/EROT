# Required modules
import numpy as np
import torch
from scipy.stats import norm
import ot


# generate marginal probability vector in \R^n supported on [lend, rend]
# resulting vector is a linear combination of normal distributions with means=locs and standard deviations=scales
def compute_marginals(lend, rend, n, locs, scales):
    x = np.linspace(lend, rend, n)
    mu = np.zeros(n)
    for (loc, scale) in zip(locs, scales):
        mu += (norm.pdf(x,loc=loc, scale=scale))/len(locs)
    mu = torch.from_numpy(mu / mu.sum())
    return mu

# Euclidean cost computes the Euclidean distance between two coordinates
def compute_euclidean_cost(n):
    print("Compute Euclidean Cost...")
    # Initialize cost matrix
    x = torch.arange(n, dtype=torch.float64) # vector in \R^n of the form [1,...,n]
    C = ot.dist(x.reshape((n,1)), x.reshape((n,1))) # Euclidean metric as a cost function
    return C/C.max() # normalize the cost

# Weak Coulomb cost sets relatively large real value for diagonal entries
def compute_weak_coulomb_cost(n):
    print("Computing Weak Coulomb Cost...")
    x = np.arange(n, dtype=np.float64) # vector in \R^n of the form [1,...,n]
    # L1 metric with diagonal entries of 1s
    C = torch.from_numpy(ot.dist(x.reshape((n,1)), x.reshape((n,1)), metric='cityblock')) + torch.diag(torch.ones(n))
    C = torch.pow(C,-1) + torch.diag((n+1)*torch.ones(n)) # element-wise inverse and take extreme values for diagonal entries
    return C/C.max() # normalize the cost

# Strong Coulomb cost sets diagonal entires to be positive infinity
def compute_strong_coulomb_cost(n):
    print("Computing Strong Coulomb Cost...")
    x = np.arange(n, dtype=np.float64) # vector in \R^n of the form [1,...,n]
    # L1 metric with diagonal entries of 1s
    C = torch.from_numpy(ot.dist(x.reshape((n,1)), x.reshape((n,1)), metric='cityblock')) + torch.diag(torch.ones(n))
    C = torch.pow(C,-1) + torch.diag(torch.ones(n) * float('inf')) # element-wise inverse and take + infinity for diagonal entries
    return C