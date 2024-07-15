# EROT
EROT is a package of numerical algorithms for Entropy Regularized Optimal Transport Problems.

</br>

## Quick Start
We provide a demo to show how to use EROT for classical quadratic entropy regularized optimal transport problem. Please follow the command lines below to try it out:

```bash

# run algorithms to compute the optimal coupling under specified entropic regularization.
python3 ./eot/run_eot.py --ot classical --entropy quadratic --cost ./demo/C.pt --marginal ./demo/X.pt ./demo/Y.pt --epsilon 1 --num_iter 50000 --error 1e-8 --out demo_sol.pt
```
There will be an output Pytorch tensor file `demo_sol.pt` encoding the optimal coupling matrix of the classical quadratic entropy regularized optimal transport problem of
$$
\mathrm{OT}_{\varepsilon}(X,Y) \coloneqq \inf \{ C_\epsilon(P) : P\in\Pi(X,Y)\}. 
$$
with Euclidean cost $C \in \mathbb{R}^{100\times 100}$, marginal probability vectors $X,Y \in \mathbb{R}^{100}$ with $X \sim \mathcal{N}(\mu_X=2.1,\sigma_X=1)$ and $Y \sim \mathcal{N}(\mu_Y=-3, \sigma_Y=0.75)$ under the regularization parameter $\varepsilon = 1$ and tolerance $10^{-8}$.

</br>

## Contents

- [YACHT](#yacht)
   * [Quick start](#quick-start)
   * [Installation](#installation)
   * [Usage](#usage)
      + [EROT Commands Overview](#yacht-commands-overview)
      + [EROT workflow](#yacht-workflow)
      + [Creating cost tensors (erot make_cost_tensor)](#creating-sketches-of-your-reference-database-genomes-yacht-sketch-ref)
      + [Creating marginal distributions (erot make_marginal_tensor)](#creating-sketches-of-your-sample-yacht-sketch-sample)
      + [Run the EROT algorithm](#run-the-yacht-algorithm-yacht-run)
         - [Parameters](#parameters-1)
         - [Output](#output-1)
      + [Convert EROT result to other popular output formats](#convert-yacht-result-to-other-popular-output-formats-yacht-convert)
         - [Parameters](#parameters-2)