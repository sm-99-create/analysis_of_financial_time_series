import numpy as np
from scipy.optimize import minimize

def log_likelihood(params, returns):
   """
   Log-likelihood function for normally distributed returns.

   Parameters:
   - params: [mu, sigma], where mu is the mean and sigma is the standard deviation
   - returns: array-like, observed return values

   Returns:
   - Log-likelihood value
   """
   mu, sigma = params
   if sigma <= 0:
       return -np.inf  # to avoid invalid log-likelihood for non-positive sigma

   returns = np.array(returns)
   T = len(returns)
   term1 = -0.5 * T * np.log(2 * np.pi)
   term2 = -T * np.log(sigma)
   term3 = -0.5 * np.sum(((returns - mu) ** 2) / (sigma ** 2))

   return term1 + term2 + term3

def negative_log_likelihood(params, returns):
   """
   Negative log-likelihood for minimization.
   """
   return -log_likelihood(params, returns)

def estimate_mle(returns, mu_init=0.0, sigma_init=1.0):
   """
   Estimate mu and sigma using Maximum Likelihood Estimation (MLE).

   Parameters:
   - returns: array-like, observed return values
   - mu_init: initial guess for mean
   - sigma_init: initial guess for standard deviation

   Returns:
   - Tuple (mu_mle, sigma_mle)
   """
   result = minimize(
       negative_log_likelihood,
       x0=[mu_init, sigma_init],
       args=(np.array(returns)),
       bounds=[(None, None), (1e-6, None)],  # sigma > 0
       method='L-BFGS-B'
   )

   if result.success:
       return result.x  # mu_mle, sigma_mle
   else:
       raise RuntimeError("MLE optimization failed: " + result.message)

if __name__ == "__main__":
   # Example usage
   returns = [0.01, -0.02, 0.015, -0.005, 0.003, 0.012]

   mu_mle, sigma_mle = estimate_mle(returns)
   print(f"MLE mu: {mu_mle:.6f}")
   print(f"MLE sigma: {sigma_mle:.6f}")
