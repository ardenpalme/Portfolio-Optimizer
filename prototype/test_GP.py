import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Example data - Simulating returns of two assets
np.random.seed(42)
asset1_returns = np.random.normal(0.05, 0.1, 1000)  # Asset 1: mean return 5%, std 10%
asset2_returns = np.random.normal(0.03, 0.08, 1000)  # Asset 2: mean return 3%, std 8%

# Portfolio return for given weights
def portfolio_return(w, returns1, returns2):
    return w[0] * returns1 + w[1] * returns2

# Compute Omega ratio
def omega_ratio(w, returns1, returns2, threshold=0):
    portfolio_returns = portfolio_return(w, returns1, returns2)
    excess_returns_above_threshold = portfolio_returns[portfolio_returns > threshold] - threshold
    excess_returns_below_threshold = threshold - portfolio_returns[portfolio_returns < threshold]
    
    omega = np.sum(excess_returns_above_threshold) / np.sum(excess_returns_below_threshold)
    return omega

# Constraints: Weights must sum to 1 and each weight must be between 0 and 1
def constraint_sum_to_one(w):
    return np.sum(w) - 1

def constraint_weights_non_negative(w):
    return w[0] >= 0 and w[1] >= 0

# Bounds for weights (between 0 and 1)
bounds = [(0, 1), (0, 1)]

# Initialize Gaussian Process
kernel = C(1.0, (1e-3, 1e1)) * RBF(1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Initial random samples
X_train = np.random.dirichlet([1, 1], size=5)  # Random weights that sum to 1
y_train = np.array([omega_ratio(w, asset1_returns, asset2_returns) for w in X_train])

# Fit the GP model
gp.fit(X_train, y_train)

# Acquisition function: Expected Improvement (EI)
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    sigma = sigma.reshape(-1, 1)
    
    mu_sample_opt = np.max(Y_sample)
    
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei

# Find the next point to evaluate based on the acquisition function
def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=10):
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Randomly sample starting points and optimize
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[0][0], bounds[0][1], size=dim)
        res = minimize(min_obj, x0=x0, bounds=bounds, constraints={"type": "eq", "fun": constraint_sum_to_one})

        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x.reshape(-1, 1)

# Run Bayesian Optimization loop
n_iterations = 20
for iteration in range(n_iterations):
    # Propose next point to evaluate
    next_sample = propose_location(expected_improvement, X_train, y_train, gp, bounds)

    # Evaluate the true objective function at the proposed location
    next_sample = next_sample.flatten()
    next_sample = np.clip(next_sample, 0, 1)  # Ensure weights are valid
    next_sample = next_sample / np.sum(next_sample)  # Ensure weights sum to 1
    next_value = omega_ratio(next_sample, asset1_returns, asset2_returns)
    
    # Add the new sample to the dataset
    X_train = np.vstack((X_train, next_sample))
    y_train = np.append(y_train, next_value)
    
    # Update the GP model
    gp.fit(X_train, y_train)

    # Print the best Omega ratio and corresponding weights so far
    best_idx = np.argmax(y_train)
    print(f"Iteration {iteration+1}: Best Omega Ratio = {y_train[best_idx]}, Weights = {X_train[best_idx]}")

# Plot Omega Ratio surface
x = np.linspace(0, 1, 100)
y = np.array([omega_ratio([w, 1-w], asset1_returns, asset2_returns) for w in x])

plt.plot(x, y)
plt.title('Omega Ratio vs. Portfolio Weight (w1)')
plt.xlabel('Weight of Asset 1 (w1)')
plt.ylabel('Omega Ratio')
plt.grid(True)
plt.show()
