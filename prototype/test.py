import numpy as np

# Example: Two 1x365 vectors representing the daily percentage returns of two assets
asset_1_returns = np.random.normal(0, 0.01, 365)  # Simulated returns for asset 1
asset_2_returns = np.random.normal(0, 0.01, 365)  # Simulated returns for asset 2

# Covariance calculation
cov_matrix = np.cov(asset_1_returns, asset_2_returns)

# Print the covariance matrix
print("Covariance Matrix:")
print(cov_matrix)

# Extract the covariance value between asset_1 and asset_2
covariance = cov_matrix[0, 1]
print(f"Covariance between asset 1 and asset 2: {covariance}")