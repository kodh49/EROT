
import logging
import jax.numpy as jnp
import jax 
from jax.scipy.special import logsumexp
import pandas as pd
import time
from scipy.stats import norm
from functools import partial
import jax.lax as lax

'''
2. remove_tensor_sum:

- Purpose: 
  
  This function adjusts the cost tensor by subtracting the "tensor sum" of potential vectors. 
  This step is necessary to align the cost tensor with dual potentials in the context of the optimization problem.

- Advantage: 
 
  It transforms the cost tensor into a form that is more suitable for subsequent calculations, 
  particularly for the entropic regularization approach used in the Sinkhorn algorithm.
'''

@jax.jit
def remove_tensor_sum(c, u):
    """
    Removes the tensor sum of potentials from the cost tensor.

    Args:
        c (jnp.ndarray): Cost tensor.
        u (list of jnp.ndarray): List of potential vectors.

    Returns:
        jnp.ndarray: Adjusted cost tensor after subtracting potentials.
    """
    k = len(u)
    for i in range(k):
        c -= jnp.expand_dims(u[i], axis=list(range(i)) + list(range(i + 1, k)))
        '''
        The purpose of this loop is to subtract a series of multi-dimensional arrays (formed by expanding each u[i]) from c.
        Each u[i] is reshaped so that it can be subtracted from c in a way that matches the dimensions of c.

        k = len(u): Determines the number of elements in u.
        for i in range(k):: Iterates over each element in u.
        jnp.expand_dims(u[i], axis=...): Reshapes u[i] by adding dimensions everywhere except the i-th position.
        c -= ...: Subtracts the reshaped u[i] from c element-wise, broadcasting u[i] to match c.
        This loop gradually adjusts the array c by subtracting the expanded versions of the elements in u.
        '''
    return c



'''
3. coupling_tensor:

- Purpose: 
  Computes the coupling tensor based on the given potentials and cost tensor. 
  This tensor represents the joint probability distribution of the marginals, adjusted for entropic regularization.

- Advantage:
  It encapsulates the regularized solution to the MMOT problem, providing a soft approximation to the optimal transport plan.
'''

@jax.jit
def coupling_tensor(potentials, cost_t, epsilon: jnp.float32) -> jnp.ndarray:
    """
    Computes the coupling tensor based on potentials and cost tensor.

    Args:
        potentials (list of jnp.ndarray): List of potential vectors.
        cost_t (jnp.ndarray): Cost tensor.
        epsilon (float): Regularization parameter.

    Returns:
        jnp.ndarray: Coupling tensor.


    - This calls the remove_tensor_sum function, which adjusts the cost tensor by subtracting
     the "tensor sum" of the potential vectors from it.

    - The purpose of remove_tensor_sum is to transform the cost tensor by removing contributions from the potentials
      in a way that aligns with optimal transport theory, where the cost tensor is adjusted based on dual potentials.
    
    - After adjusting the cost tensor, the function exponentiates the negative of the result divided by 
      the regularization parameter epsilon.

    - The expression jnp.exp(-X / epsilon) is typical in entropic regularization of optimal transport, 
      where the exponentiated form represents the "soft" or regularized version of the coupling matrix.
        """
    return jnp.exp(-remove_tensor_sum(cost_t, potentials) / epsilon)


for i in range(N):
    axis = list(range(i)) + list(range(i + 1, N))
    potencials_minus_c = remove_tensor_sum(c, potentials)
    lse = logsumexp(potencials_minus_c / -reg, axis=axis)
    potentials[i] += reg * jnp.log(marginals[i]) - reg * lse




import logging
import jax.numpy as jnp
import jax 
from jax.scipy.special import logsumexp
import pandas as pd
import time
from scipy.stats import norm
from functools import partial
import jax.lax as lax



'''
This code is designed to solve multimarginal optimal transport (MMOT) problems using the Sinkhorn algorithm, 
which is an iterative method for approximating the optimal coupling between multiple marginal distributions under
certain cost functions. The code is implemented using JAX for efficient computation and automatic differentiation.

'''




# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a range of values for x with float32
x = jnp.linspace(-5, 5, 100, dtype=jnp.float32)

# Define Gaussian distributions for each marginal with float32
mu_1 = jnp.asarray(norm.pdf(x, loc=1.5, scale=0.5), dtype=jnp.float32)
mu_2 = jnp.asarray(norm.pdf(x, loc=2.5, scale=0.9), dtype=jnp.float32)
mu_3 = jnp.asarray(norm.pdf(x, loc=2, scale=0.6), dtype=jnp.float32)
mu_4 = jnp.asarray(norm.pdf(x, loc=3, scale=0.8), dtype=jnp.float32)
mu_5 = jnp.asarray(norm.pdf(x, loc=2.75, scale=0.7), dtype=jnp.float32)

# Normalize the distributions
mu_1 = mu_1 / mu_1.sum()
mu_2 = mu_2 / mu_2.sum()
mu_3 = mu_3 / mu_3.sum()
mu_4 = mu_4 / mu_4.sum()
mu_5 = mu_5 / mu_5.sum()









'''
1. compute_cost_matrix_coulomb_jax & compute_cost_matrix_quadratic_jax:

- Purpose:
  These functions compute the cost matrices for MMOT problems,
  where the cost is either based on the Coulomb interaction or a quadratic function.
  The cost matrices are multi-dimensional tensors representing the interaction costs
  between different elements in the marginals.


- Difference:
  The Coulomb cost uses an inverse distance metric (1/distance), 
  while the quadratic cost uses the squared distance (distance^2).

- Advantage: 
  Both are important in different contexts. 
  The Coulomb cost is suitable for physical systems like charged particles, 
  while the quadratic cost is often used in more general optimal transport problems.
'''

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_coulomb_jax(x, N):
    """
    Computes the Coulomb cost matrix for N marginals using JAX.

    Args:
        x (jnp.ndarray): Input array of shape (n,).
        N (int): Number of marginals.

    Returns:
        jnp.ndarray: Coulomb cost matrix of shape (n, n, ..., n) for N dimensions.
    """
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns, dtype=jnp.float32)

    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, jnp.inf)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            '''
            The inner vmap computes a vector of Coulomb pairwise interactions between a fixed x_ and every other element y_ in x.
            The outer vmap repeats this process for each x_ in x, resulting in a matrix cost_m, where each element represents the pairwise 
            interaction between different elements of x.
            '''
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            '''
            The purpose of this code is to create a list of axis indices that excludes i and j. 
            This is commonly used in operations where you want to sum or manipulate data across all dimensions except for specific ones.
            '''
            total_cost += jnp.expand_dims(cost_m, axis=axis)
            '''
            jnp.expand_dims: Adds new dimensions to cost_m at the specified axis positions.
            axis=axis: Specifies where to insert these new dimensions.
            total_cost += ...: Adds the expanded cost_m to total_cost, leveraging broadcasting to match shapes.
            This operation allows cost_m to be added to total_cost correctly, even if their shapes initially differ, 
            by expanding cost_m to have compatible dimensions.
            '''

    return total_cost

coulomb_2m = compute_cost_matrix_coulomb_jax(x,2)
logging.info(f"Successfully computed coulomb_2m.")
logging.info(f"Shape of coulomb_2m: {coulomb_2m.shape}")

coulomb_3m = compute_cost_matrix_coulomb_jax(x,3)
logging.info(f"Successfully computed coulomb_3m.")
logging.info(f"Shape of coulomb_3m: {coulomb_3m.shape}")

coulomb_4m = compute_cost_matrix_coulomb_jax(x,4)
logging.info(f"Successfully computed coulomb_4m.")
logging.info(f"Shape of coulomb_4m: {coulomb_4m.shape}")

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_quadratic_jax(x, N):
    """
    Computes the Quadratic cost matrix for N marginals using JAX.

    Args:
        x (jnp.ndarray): Input array of shape (n,).
        N (int): Number of marginals.

    Returns:
        jnp.ndarray: Quadratic cost matrix of shape (n, n, ..., n) for N dimensions.
    """
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns, dtype=jnp.float32)

    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, diff**2, 0)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)

    return total_cost

quadratic_2m = compute_cost_matrix_quadratic_jax(x,2)
logging.info(f"Successfully computed quadratic_2m.")
logging.info(f"Shape of quadratic_2m: {quadratic_2m.shape}")

quadratic_3m = compute_cost_matrix_quadratic_jax(x,3)
logging.info(f"Successfully computed quadratic_3m.")
logging.info(f"Shape of quadratic_3m: {quadratic_3m.shape}")

quadratic_4m = compute_cost_matrix_quadratic_jax(x,4)
logging.info(f"Successfully computed quadratic_4m.")
logging.info(f"Shape of quadratic_4m: {quadratic_4m.shape}")





def preprocess_cost_tensor(c, N):
    """
    Preprocess the cost tensor by splitting it into the correct coordinates.

    Args:
        c (jnp.ndarray): Cost tensor.
        N (int): Number of potentials.

    Returns:
        list of jnp.ndarray: Preprocessed tensors for each potential.
    """
    preprocessed_c = []
    for i in range(N):
        axis = list(range(i)) + list(range(i + 1, N))
        preprocessed_c.append(jnp.expand_dims(c, axis=axis))
    return preprocessed_c













@jax.jit
def coupling_tensor(potentials, preprocessed_c, epsilon: jnp.float32) -> jnp.ndarray:
    """
    Computes the coupling tensor based on potentials and preprocessed cost tensor.

    Args:
        potentials (list of jnp.ndarray): List of potential vectors.
        preprocessed_c (list of jnp.ndarray): List of preprocessed cost tensors.
        epsilon (float): Regularization parameter.

    Returns:
        jnp.ndarray: Coupling tensor.
    """
    # Compute the tensor after subtracting potentials from preprocessed cost tensors
    result = []
    for i, potential in enumerate(potentials):
        tensor_minus_potential = preprocessed_c[i] - potential
        result.append(tensor_minus_potential)

    # Combine results to form the coupling tensor
    # Here, we assume that the `result` tensors can be combined appropriately
    coupling = jnp.sum(jnp.stack(result), axis=0)  # Adjust combination as needed
    
    return jnp.exp(-coupling / epsilon)















'''
4. tensor_marginal & tensor_marginals:

- Purpose: 
  These functions compute marginal distributions from the coupling tensor by summing over all dimensions except the one of interest.
  The result is the marginal distribution corresponding to that dimension.

- Advantage: 
  These are essential for comparing the computed marginals with the target marginals to check the accuracy of the solution.
'''
def tensor_marginal(coupling: jnp.ndarray, slice_index: int) -> jnp.ndarray:
    """
    Computes a marginal of the coupling tensor by summing over all but one dimension.

    The tensor_marginal function calculates a marginal distribution from a multi-dimensional 
    coupling tensor by summing over all dimensions except one specified dimension. 

    This is often done in the context of optimal transport or probability distributions,
     where marginal distributions are needed.

    Args:
        coupling (jnp.ndarray): Coupling tensor.
        slice_index (int): Index of the dimension to keep.

    Returns:
        jnp.ndarray: Computed marginal distribution.
    """
    
    k = coupling.ndim
    axis = list(range(slice_index)) + list(range(slice_index + 1, k))

    '''
    coupling.ndim gives the number of dimensions (or rank) of the coupling tensor. The variable k represents this number.

    This line constructs a list axis containing all dimension indices except the one specified by slice_index.

    range(slice_index): Generates a list of integers from 0 to slice_index - 1.

    range(slice_index + 1, k): Generates a list of integers from slice_index + 1 to k - 1.

    Concatenation: The two lists are concatenated, creating a list of indices that includes all dimensions except the slice_index dimension.
    
    coupling.sum(axis=axis): Sums over all dimensions specified in axis.

    Since axis contains all dimensions except the one at slice_index, the sum is performed over all other dimensions,
    effectively collapsing them and leaving only the dimension at slice_index.

    The result is a lower-dimensional array (tensor) that represents the marginal distribution
    of the coupling tensor along the specified slice_index.
        
    '''
    return coupling.sum(axis=axis)

@jax.jit
def tensor_marginals(tensor):
     """
    Computes all marginals of the coupling tensor.

    Args:
        tensor (jnp.ndarray): Coupling tensor.

    Returns:
        tuple: Tuple of computed marginals.
    
    
    
    range(tensor.ndim):

    tensor.ndim: This returns the number of dimensions (or axes) of the tensor.
    range(tensor.ndim): Generates a sequence of integers from 0 to tensor.ndim - 1, 
    representing all possible dimension indices of the tensor.
    for ix in range(tensor.ndim):

    This is a generator expression that iterates over all dimension indices of the tensor.
    ix: Each ix represents a dimension index, ranging from 0 to tensor.ndim - 1.
    tensor_marginal(tensor, ix):

    For each dimension index ix, the function tensor_marginal(tensor, ix) is called.
    tensor_marginal(tensor, ix): Computes the marginal distribution of the tensor along 
    the ix-th dimension. It sums over all dimensions of the tensor except the ix-th dimension,
        returning a reduced-dimensional tensor (or array).

    tuple(...):

    The tuple() function takes the generator expression and converts it into a tuple.
    Each element of the tuple is the marginal distribution of the tensor along one of its dimensions.

     """
     return tuple(tensor_marginal(tensor, ix) for ix in range(tensor.ndim))














'''
5. compute_error:

    - Purpose:
      Calculates the error between the computed marginals and the given marginals, 
      which helps in assessing the convergence of the Sinkhorn algorithm.
    
    - Advantage: 
      Provides a quantitative measure of how close the computed solution is to the desired one, guiding the iterative process.
'''
@jax.jit
def compute_error(potentials, marginals, cost, epsilon):
    """
    Computes the error between the computed marginals and the given marginals.

    Args:
        potentials (list of jnp.ndarray): List of potential vectors.
        marginals (list of jnp.ndarray): List of target marginal distributions.
        cost (jnp.ndarray): Cost tensor.
        epsilon (float): Regularization parameter.

    Returns:
        jnp.ndarray: Array of errors for each marginal.
    """
    coupl_tensor = coupling_tensor(potentials, cost, epsilon)
    computed_marginals = tensor_marginals(coupl_tensor)
    errors = jnp.array([
        jnp.sum(jnp.abs(marginal - computed_marginal))
        for marginal, computed_marginal in zip(marginals, computed_marginals)
    ])
    return errors















'''
6. sinkhorn_logsumexp:

- Purpose: 
   This is the core function that iteratively updates the potentials using the log-sum-exp trick to solve the MMOT problem. 
   It also logs the number of steps, time taken, and error at each iteration.

- Advantage: 
   The Sinkhorn algorithm is highly efficient, especially when using JAX,
   which allows for automatic differentiation and GPU acceleration. The log-sum-exp trick helps in maintaining numerical stability.
'''

def sinkhorn_logsumexp(marginals, c, reg, precision=1e-3, max_iters=350):
    """
    Solves the multimarginal optimal transport problem using the Sinkhorn algorithm.

    Args:
        marginals (list of jnp.ndarray): List of N marginal distributions, each of shape (n,).
        c (jnp.ndarray): Cost tensor of shape (n, n, ..., n) for N dimensions.
        reg (float): Regularization parameter.
        precision (float, optional): Desired precision for convergence. Default is 1e-3. 
        max_iters (int, optional): Maximum number of iterations. Default is 10000.

    Returns:
        tuple: A tuple containing:
            - P (jnp.ndarray): Computed joint probability distribution of shape (n, n, ..., n).
            - log_data (dict): Log data containing the number of steps, time taken, and errors.
    """
    # Ensure precision and reg are float32
    precision = jnp.float32(precision)
    reg = jnp.float32(reg)
    logging.info("Starting Sinkhorn algorithm")
    start_time = time.time()
    N = len(marginals)
    n = marginals[0].shape[0]

    potentials = [jnp.zeros(n, jnp.float32) for i in range(N)]
    logging.info(f"Initialized potentials with shape {n} for {N} marginals")
    preprocessed_c = preprocess_cost_tensor(c, N)


    def body_fn(var):
        (potentials, iter, errors) = var

        errors = jax.lax.cond(
            iter % 200 == 0,
            lambda: compute_error(potentials, marginals, c, reg),
            lambda: errors
        )

        iter += 1


        for i in range(N):
            # Compute the coupling tensor
            coupling = coupling_tensor(potentials, preprocessed_c, reg)
            
            # Update potentials
            axis = list(range(i)) + list(range(i + 1, N))
            lse = jax.scipy.special.logsumexp(coupling / -reg, axis=axis)
            potentials[i] += reg* jnp.log(marginals[i]) - reg * lse
            '''
            The code iterates over each dimension to update the corresponding potential vector by 
            considering the log-sum-exp of the adjusted cost tensor and the logarithm of the marginals.

            The regularization parameter reg is used to scale these updates, and the process aims to gradually 
            adjust the potentials to satisfy certain optimality conditions in an entropic regularized optimization problem.

            The final result is a set of updated potentials that are closer to the solution of the problem.
            '''
        
        return potentials, iter, errors


    potentials, iter, errors = jax.lax.while_loop(
        lambda var: jnp.logical_and(jnp.max(var[2]) > precision, var[1] <= max_iters),
        lambda var: body_fn(var),
        (potentials, 0, jnp.full(N, jnp.inf))
    )

    P = coupling_tensor(potentials, c, reg)

    log_data = {
        'steps': iter,
        'time': time.time() - start_time,
        'errors': errors,
    }
    
    logging.info(f"Completed Sinkhorn algorithm in {log_data['steps']} steps and {log_data['time']:.2f} seconds")
    logging.info(f"Final errors: {log_data['errors']}")

    return P, log_data

# Define the regularization parameters, marginals, and cost matrices
regularization = [10]
marginals = [mu_1, mu_2, mu_3, mu_4]
coulomb_costs = [coulomb_2m, coulomb_3m, coulomb_4m]
quadratic_costs = [quadratic_2m, quadratic_3m, quadratic_4m]

# Create empty DataFrames for each case
df_2marginals = pd.DataFrame()
df_3marginals = pd.DataFrame()
df_4marginals = pd.DataFrame()

# Loop over the number of marginals
for i in range(2, 5):  
    marginals_subset = marginals[:i]
    if i == 2:
        cost_matrices = [(coulomb_costs[0], quadratic_costs[0])]
    elif i == 3:
        cost_matrices = [(coulomb_costs[1], quadratic_costs[1])]
    elif i == 4:
        cost_matrices = [(coulomb_costs[2], quadratic_costs[2])]

    results = []
    for reg in regularization:
        for coulomb_cost, quadratic_cost in cost_matrices:
            logging.info(f"Processing: {i} marginals, Coulomb cost, regularization = {reg}")
            # Run Sinkhorn algorithm for Coulomb cost
            P_coulomb, log_data_coulomb = sinkhorn_logsumexp(marginals_subset, coulomb_cost, reg)
            results.append({
                'regularization': reg,
                'cost_type': 'Coulomb',
                'marginals': i,
                'steps': log_data_coulomb['steps'],
                'time': log_data_coulomb['time'],
                'errors': log_data_coulomb['errors'].tolist()
            })

            logging.info(f"Processing: {i} marginals, Quadratic cost, regularization = {reg}")
            # Run Sinkhorn algorithm for Quadratic cost
            P_quadratic, log_data_quadratic = sinkhorn_logsumexp(marginals_subset, quadratic_cost, reg)
            results.append({
                'regularization': reg,
                'cost_type': 'Quadratic',
                'marginals': i,
                'steps': log_data_quadratic['steps'],
                'time': log_data_quadratic['time'],
                'errors': log_data_quadratic['errors'].tolist()
            })

    # Convert the results list into a DataFrame and store it in the corresponding DataFrame
    if i == 2:
        df_2marginals = pd.DataFrame(results)
    elif i == 3:
        df_3marginals = pd.DataFrame(results)
    elif i == 4:
        df_4marginals = pd.DataFrame(results)


logging.info('Dataset')
df_2marginals.to_csv('df_2marginal.csv', index=False)
df_3marginals.to_csv('df_3marginal.csv', index=False)
df_4marginals.to_csv('df_4marginal.csv', index=False)









import logging
import jax.numpy as jnp
import jax 
from jax.scipy.special import logsumexp
import pandas as pd
import time
from scipy.stats import norm
from functools import partial
import jax.lax as lax



'''
This code is designed to solve multimarginal optimal transport (MMOT) problems using the Sinkhorn algorithm, 
which is an iterative method for approximating the optimal coupling between multiple marginal distributions under
certain cost functions. The code is implemented using JAX for efficient computation and automatic differentiation.

'''




# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a range of values for x with float32
x = jnp.linspace(-5, 5, 100, dtype=jnp.float32)

# Define Gaussian distributions for each marginal with float32
mu_1 = jnp.asarray(norm.pdf(x, loc=1.5, scale=0.5), dtype=jnp.float32)
mu_2 = jnp.asarray(norm.pdf(x, loc=2.5, scale=0.9), dtype=jnp.float32)
mu_3 = jnp.asarray(norm.pdf(x, loc=2, scale=0.6), dtype=jnp.float32)
mu_4 = jnp.asarray(norm.pdf(x, loc=3, scale=0.8), dtype=jnp.float32)
mu_5 = jnp.asarray(norm.pdf(x, loc=2.75, scale=0.7), dtype=jnp.float32)

# Normalize the distributions
mu_1 = mu_1 / mu_1.sum()
mu_2 = mu_2 / mu_2.sum()
mu_3 = mu_3 / mu_3.sum()
mu_4 = mu_4 / mu_4.sum()
mu_5 = mu_5 / mu_5.sum()









'''
1. compute_cost_matrix_coulomb_jax & compute_cost_matrix_quadratic_jax:

- Purpose:
  These functions compute the cost matrices for MMOT problems,
  where the cost is either based on the Coulomb interaction or a quadratic function.
  The cost matrices are multi-dimensional tensors representing the interaction costs
  between different elements in the marginals.


- Difference:
  The Coulomb cost uses an inverse distance metric (1/distance), 
  while the quadratic cost uses the squared distance (distance^2).

- Advantage: 
  Both are important in different contexts. 
  The Coulomb cost is suitable for physical systems like charged particles, 
  while the quadratic cost is often used in more general optimal transport problems.
'''

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_coulomb_jax(x, N):
    """
    Computes the Coulomb cost matrix for N marginals using JAX.

    Args:
        x (jnp.ndarray): Input array of shape (n,).
        N (int): Number of marginals.

    Returns:
        jnp.ndarray: Coulomb cost matrix of shape (n, n, ..., n) for N dimensions.
    """
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns, dtype=jnp.float32)

    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, 1 / diff, jnp.inf)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            '''
            The inner vmap computes a vector of Coulomb pairwise interactions between a fixed x_ and every other element y_ in x.
            The outer vmap repeats this process for each x_ in x, resulting in a matrix cost_m, where each element represents the pairwise 
            interaction between different elements of x.
            '''
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            '''
            The purpose of this code is to create a list of axis indices that excludes i and j. 
            This is commonly used in operations where you want to sum or manipulate data across all dimensions except for specific ones.
            '''
            total_cost += jnp.expand_dims(cost_m, axis=axis)
            '''
            jnp.expand_dims: Adds new dimensions to cost_m at the specified axis positions.
            axis=axis: Specifies where to insert these new dimensions.
            total_cost += ...: Adds the expanded cost_m to total_cost, leveraging broadcasting to match shapes.
            This operation allows cost_m to be added to total_cost correctly, even if their shapes initially differ, 
            by expanding cost_m to have compatible dimensions.
            '''

    return total_cost

coulomb_2m = compute_cost_matrix_coulomb_jax(x,2)
logging.info(f"Successfully computed coulomb_2m.")
logging.info(f"Shape of coulomb_2m: {coulomb_2m.shape}")

coulomb_3m = compute_cost_matrix_coulomb_jax(x,3)
logging.info(f"Successfully computed coulomb_3m.")
logging.info(f"Shape of coulomb_3m: {coulomb_3m.shape}")

coulomb_4m = compute_cost_matrix_coulomb_jax(x,4)
logging.info(f"Successfully computed coulomb_4m.")
logging.info(f"Shape of coulomb_4m: {coulomb_4m.shape}")

@partial(jax.jit, static_argnums=[1])
def compute_cost_matrix_quadratic_jax(x, N):
    """
    Computes the Quadratic cost matrix for N marginals using JAX.

    Args:
        x (jnp.ndarray): Input array of shape (n,).
        N (int): Number of marginals.

    Returns:
        jnp.ndarray: Quadratic cost matrix of shape (n, n, ..., n) for N dimensions.
    """
    n = x.shape[0]
    ns = (n,) * N
    total_cost = jnp.zeros(ns, dtype=jnp.float32)

    def coulumb_pairwise(x, y):
        diff = jnp.abs(x - y)
        return jnp.where(diff != 0, diff**2, 0)

    for i in range(N):
        for j in range(i + 1, N):
            cost_m = jax.vmap(lambda x_: jax.vmap(lambda y_: coulumb_pairwise(x_, y_))(x))(x)
            axis = list(range(i)) + list(range(i+1, j)) + list(range(j + 1, N))
            total_cost += jnp.expand_dims(cost_m, axis=axis)

    return total_cost

quadratic_2m = compute_cost_matrix_quadratic_jax(x,2)
logging.info(f"Successfully computed quadratic_2m.")
logging.info(f"Shape of quadratic_2m: {quadratic_2m.shape}")

quadratic_3m = compute_cost_matrix_quadratic_jax(x,3)
logging.info(f"Successfully computed quadratic_3m.")
logging.info(f"Shape of quadratic_3m: {quadratic_3m.shape}")

quadratic_4m = compute_cost_matrix_quadratic_jax(x,4)
logging.info(f"Successfully computed quadratic_4m.")
logging.info(f"Shape of quadratic_4m: {quadratic_4m.shape}")





def preprocess_cost_tensor(c, N):
    """
    Preprocess the cost tensor by splitting it into the correct coordinates.

    Args:
        c (jnp.ndarray): Cost tensor.
        N (int): Number of potentials.

    Returns:
        list of jnp.ndarray: Preprocessed tensors for each potential.
    """
    preprocessed_c = []
    for i in range(N):
        axis = list(range(i)) + list(range(i + 1, N))
        preprocessed_c.append(jnp.expand_dims(c, axis=axis))
    return preprocessed_c













@jax.jit
def coupling_tensor(potentials, preprocessed_c, epsilon: jnp.float32) -> jnp.ndarray:
    """
    Computes the coupling tensor based on potentials and preprocessed cost tensor.

    Args:
        potentials (list of jnp.ndarray): List of potential vectors.
        preprocessed_c (list of jnp.ndarray): List of preprocessed cost tensors.
        epsilon (float): Regularization parameter.

    Returns:
        jnp.ndarray: Coupling tensor.
    """
    # Compute the tensor after subtracting potentials from preprocessed cost tensors
    result = []
    for i, potential in enumerate(potentials):
        tensor_minus_potential = preprocessed_c[i] - potential
        result.append(tensor_minus_potential)

    # Combine results to form the coupling tensor
    # Here, we assume that the `result` tensors can be combined appropriately
    coupling = jnp.sum(jnp.stack(result), axis=0)  # Adjust combination as needed
    
    return jnp.exp(-coupling / epsilon)















'''
4. tensor_marginal & tensor_marginals:

- Purpose: 
  These functions compute marginal distributions from the coupling tensor by summing over all dimensions except the one of interest.
  The result is the marginal distribution corresponding to that dimension.

- Advantage: 
  These are essential for comparing the computed marginals with the target marginals to check the accuracy of the solution.
'''
def tensor_marginal(coupling: jnp.ndarray, slice_index: int) -> jnp.ndarray:
    """
    Computes a marginal of the coupling tensor by summing over all but one dimension.

    The tensor_marginal function calculates a marginal distribution from a multi-dimensional 
    coupling tensor by summing over all dimensions except one specified dimension. 

    This is often done in the context of optimal transport or probability distributions,
     where marginal distributions are needed.

    Args:
        coupling (jnp.ndarray): Coupling tensor.
        slice_index (int): Index of the dimension to keep.

    Returns:
        jnp.ndarray: Computed marginal distribution.
    """
    
    k = coupling.ndim
    axis = list(range(slice_index)) + list(range(slice_index + 1, k))

    '''
    coupling.ndim gives the number of dimensions (or rank) of the coupling tensor. The variable k represents this number.

    This line constructs a list axis containing all dimension indices except the one specified by slice_index.

    range(slice_index): Generates a list of integers from 0 to slice_index - 1.

    range(slice_index + 1, k): Generates a list of integers from slice_index + 1 to k - 1.

    Concatenation: The two lists are concatenated, creating a list of indices that includes all dimensions except the slice_index dimension.
    
    coupling.sum(axis=axis): Sums over all dimensions specified in axis.

    Since axis contains all dimensions except the one at slice_index, the sum is performed over all other dimensions,
    effectively collapsing them and leaving only the dimension at slice_index.

    The result is a lower-dimensional array (tensor) that represents the marginal distribution
    of the coupling tensor along the specified slice_index.
        
    '''
    return coupling.sum(axis=axis)

@jax.jit
def tensor_marginals(tensor):
     """
    Computes all marginals of the coupling tensor.

    Args:
        tensor (jnp.ndarray): Coupling tensor.

    Returns:
        tuple: Tuple of computed marginals.
    
    
    
    range(tensor.ndim):

    tensor.ndim: This returns the number of dimensions (or axes) of the tensor.
    range(tensor.ndim): Generates a sequence of integers from 0 to tensor.ndim - 1, 
    representing all possible dimension indices of the tensor.
    for ix in range(tensor.ndim):

    This is a generator expression that iterates over all dimension indices of the tensor.
    ix: Each ix represents a dimension index, ranging from 0 to tensor.ndim - 1.
    tensor_marginal(tensor, ix):

    For each dimension index ix, the function tensor_marginal(tensor, ix) is called.
    tensor_marginal(tensor, ix): Computes the marginal distribution of the tensor along 
    the ix-th dimension. It sums over all dimensions of the tensor except the ix-th dimension,
        returning a reduced-dimensional tensor (or array).

    tuple(...):

    The tuple() function takes the generator expression and converts it into a tuple.
    Each element of the tuple is the marginal distribution of the tensor along one of its dimensions.

     """
     return tuple(tensor_marginal(tensor, ix) for ix in range(tensor.ndim))














'''
5. compute_error:

    - Purpose:
      Calculates the error between the computed marginals and the given marginals, 
      which helps in assessing the convergence of the Sinkhorn algorithm.
    
    - Advantage: 
      Provides a quantitative measure of how close the computed solution is to the desired one, guiding the iterative process.
'''
@jax.jit
def compute_error(potentials, marginals, cost, epsilon):
    """
    Computes the error between the computed marginals and the given marginals.

    Args:
        potentials (list of jnp.ndarray): List of potential vectors.
        marginals (list of jnp.ndarray): List of target marginal distributions.
        cost (jnp.ndarray): Cost tensor.
        epsilon (float): Regularization parameter.

    Returns:
        jnp.ndarray: Array of errors for each marginal.
    """
    coupl_tensor = coupling_tensor(potentials, cost, epsilon)
    computed_marginals = tensor_marginals(coupl_tensor)
    errors = jnp.array([
        jnp.sum(jnp.abs(marginal - computed_marginal))
        for marginal, computed_marginal in zip(marginals, computed_marginals)
    ])
    return errors















'''
6. sinkhorn_logsumexp:

- Purpose: 
   This is the core function that iteratively updates the potentials using the log-sum-exp trick to solve the MMOT problem. 
   It also logs the number of steps, time taken, and error at each iteration.

- Advantage: 
   The Sinkhorn algorithm is highly efficient, especially when using JAX,
   which allows for automatic differentiation and GPU acceleration. The log-sum-exp trick helps in maintaining numerical stability.
'''

def sinkhorn_logsumexp(marginals, c, reg, precision=1e-3, max_iters=350):
    """
    Solves the multimarginal optimal transport problem using the Sinkhorn algorithm.

    Args:
        marginals (list of jnp.ndarray): List of N marginal distributions, each of shape (n,).
        c (jnp.ndarray): Cost tensor of shape (n, n, ..., n) for N dimensions.
        reg (float): Regularization parameter.
        precision (float, optional): Desired precision for convergence. Default is 1e-3. 
        max_iters (int, optional): Maximum number of iterations. Default is 10000.

    Returns:
        tuple: A tuple containing:
            - P (jnp.ndarray): Computed joint probability distribution of shape (n, n, ..., n).
            - log_data (dict): Log data containing the number of steps, time taken, and errors.
    """
    # Ensure precision and reg are float32
    precision = jnp.float32(precision)
    reg = jnp.float32(reg)
    logging.info("Starting Sinkhorn algorithm")
    start_time = time.time()
    N = len(marginals)
    n = marginals[0].shape[0]

    potentials = [jnp.zeros(n, jnp.float32) for i in range(N)]
    logging.info(f"Initialized potentials with shape {n} for {N} marginals")
    preprocessed_c = preprocess_cost_tensor(c, N)


    def body_fn(var):
        (potentials, iter, errors) = var

        # Compute errors only if necessary
        new_errors = jax.lax.cond(
            iter % 200 == 0,
            lambda: compute_error(potentials, marginals, c, reg),
            lambda: errors
        )

        iter += 1


        for i in range(N):
            # Compute the coupling tensor
            coupling = coupling_tensor(potentials, preprocessed_c, reg)
            
            # Update potentials
            axis = list(range(i)) + list(range(i + 1, N))
            lse = jax.scipy.special.logsumexp(coupling / -reg, axis=axis)
            potentials[i] += reg * jnp.log(marginals[i]) - reg * lse
            '''
            The code iterates over each dimension to update the corresponding potential vector by 
            considering the log-sum-exp of the adjusted cost tensor and the logarithm of the marginals.

            The regularization parameter reg is used to scale these updates, and the process aims to gradually 
            adjust the potentials to satisfy certain optimality conditions in an entropic regularized optimization problem.

            The final result is a set of updated potentials that are closer to the solution of the problem.
            '''
        
        return potentials, iter, new_errors 


    potentials, iter, new_errors = jax.lax.while_loop(
        lambda var: jnp.logical_and(jnp.max(var[2]) > precision, var[1] <= max_iters),
        lambda var: body_fn(var),
        (potentials, 0, jnp.full(N, jnp.inf))
    )

    P = coupling_tensor(potentials, c, reg)

    log_data = {
        'steps': iter,
        'time': time.time() - start_time,
        'errors': new_errors,
    }
    
    logging.info(f"Completed Sinkhorn algorithm in {log_data['steps']} steps and {log_data['time']:.2f} seconds")
    logging.info(f"Final errors: {log_data['errors']}")

    return P, log_data

# Define the regularization parameters, marginals, and cost matrices
regularization = [10]
marginals = [mu_1, mu_2, mu_3, mu_4]
coulomb_costs = [coulomb_2m, coulomb_3m, coulomb_4m]
quadratic_costs = [quadratic_2m, quadratic_3m, quadratic_4m]

# Create empty DataFrames for each case
df_2marginals = pd.DataFrame()
df_3marginals = pd.DataFrame()
df_4marginals = pd.DataFrame()

# Loop over the number of marginals
for i in range(2, 5):  
    marginals_subset = marginals[:i]
    if i == 2:
        cost_matrices = [(coulomb_costs[0], quadratic_costs[0])]
    elif i == 3:
        cost_matrices = [(coulomb_costs[1], quadratic_costs[1])]
    elif i == 4:
        cost_matrices = [(coulomb_costs[2], quadratic_costs[2])]

    results = []
    for reg in regularization:
        for coulomb_cost, quadratic_cost in cost_matrices:
            logging.info(f"Processing: {i} marginals, Coulomb cost, regularization = {reg}")
            # Run Sinkhorn algorithm for Coulomb cost
            P_coulomb, log_data_coulomb = sinkhorn_logsumexp(marginals_subset, coulomb_cost, reg)
            results.append({
                'regularization': reg,
                'cost_type': 'Coulomb',
                'marginals': i,
                'steps': log_data_coulomb['steps'],
                'time': log_data_coulomb['time'],
                'errors': log_data_coulomb['errors'].tolist()
            })

            logging.info(f"Processing: {i} marginals, Quadratic cost, regularization = {reg}")
            # Run Sinkhorn algorithm for Quadratic cost
            P_quadratic, log_data_quadratic = sinkhorn_logsumexp(marginals_subset, quadratic_cost, reg)
            results.append({
                'regularization': reg,
                'cost_type': 'Quadratic',
                'marginals': i,
                'steps': log_data_quadratic['steps'],
                'time': log_data_quadratic['time'],
                'errors': log_data_quadratic['errors'].tolist()
            })

    # Convert the results list into a DataFrame and store it in the corresponding DataFrame
    if i == 2:
        df_2marginals = pd.DataFrame(results)
    elif i == 3:
        df_3marginals = pd.DataFrame(results)
    elif i == 4:
        df_4marginals = pd.DataFrame(results)


logging.info('Dataset')
df_2marginals.to_csv('df_2marginal.csv', index=False)
df_3marginals.to_csv('df_3marginal.csv', index=False)
df_4marginals.to_csv('df_4marginal.csv', index=False)
