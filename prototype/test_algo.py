import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
import requests
import numpy as np
import pandas as pd
import datetime
import os

TRADING_DAYS = 365

class MarketData:
    def __init__(self, ticker, num_historical_days):
        self.ticker = ticker            
        self.price_return = None         
        self.geom_mean = None           
        self.mean_return = None           
        self.std_return = None           
        self.api_key = os.getenv('POLYGON_API_KEY')  
        self.num_historical_days = num_historical_days

        if not self.api_key:
            raise ValueError("API key not found. Please set the 'POLYGON_API_KEY' environment variable.")

    def get_data(self):
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=self.num_historical_days)

        url = "https://api.polygon.io/v2/aggs/ticker/X:{}/range/1/day/{}/{}?".format(
            self.ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apiKey': self.api_key 
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None

        data = response.json()
        close_prices = [result['c'] for result in data['results']]
        price_series = np.array(close_prices)

        price_return = []
        prev_close = 0.0
        for close in price_series:
            if prev_close == 0:
                prev_close = close
                continue

            price_return.append((close - prev_close) / prev_close) # percentage return
            prev_close = close

        self.price_return = np.array(price_return)
        self.mean_return = np.mean(self.price_return)
        self.std_return = np.std(self.price_return)

        #@assert(self.num_historical_days == len(self.price_return))
        cumulative_return = np.prod(1 + self.price_return)  
        self.geom_mean = (cumulative_return ** (TRADING_DAYS / len(self.price_return))) - 1

    def plot_returns(self, ax):
        counts, bins, _ = ax.hist(self.price_return, bins=100, density=True, alpha=0.6, color='b')

        normal_dist_x = np.linspace(bins[0], bins[-1], 1000)
        normal_dist_y = norm.pdf(normal_dist_x, self.mean_return, self.std_return)
        ax.plot(normal_dist_x, normal_dist_y, color='red', label='Normal Distribution')

        ax.axvline(self.mean_return, color='red', linestyle='--', label=f'Mean: {self.mean_return:.4f}')

        ax.text(0.05, 0.95, f'CAGR: {self.geom_mean:.4%}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top')

        ax.set_title("{} Price Returns".format(self.ticker))
        ax.set_xlabel('Percentage Return')
        ax.set_ylabel('Relative Frequency')
        ax.grid(True)
        ax.legend()

    def get_perf_ratio(self):
        return self.mean_return / self.std_return
           
if __name__ == "__main__":
    # Load asset data (e.g., daily price returns)
    asset_1 = MarketData("BTCUSD", 3000)
    asset_2 = MarketData("ETHUSD", 3000)

    asset_1.get_data()
    asset_2.get_data()

    # Stack daily asset returns into a 2D array (each row is an asset, each column is a daily return)
    asset_returns = np.array([asset_1.price_return, asset_2.price_return])
    
    # Convert to torch tensor for gradient-based optimization
    price_returns = torch.tensor(asset_returns, dtype=torch.float32, requires_grad=False)

    # Initial portfolio weights (random initialization)
    w = torch.tensor(np.random.rand(1, 2), dtype=torch.float32, requires_grad=True)

    # Set the learning rate for manual updates
    learning_rate = 0.01

    # Set risk-free rate (e.g., 0 for simplicity)
    rf = 0.0  # Can be updated to actual risk-free rate if needed

    # Number of iterations for optimization
    iterations = 1000

    for i in range(iterations):
        # Zero out gradients from previous iteration
        if w.grad is not None:
            w.grad.zero_()

        # Normalize weights to ensure they sum to 1
        w_normalized = torch.nn.functional.softmax(w, dim=1)

        # Portfolio returns calculation (weighted sum of individual asset returns)
        rp = torch.matmul(w_normalized, price_returns)

        # Calculate mean and standard deviation of portfolio returns (daily data)
        mean_rp = torch.mean(rp)
        std_rp = torch.std(rp)

        # Proper Sharpe ratio calculation (daily data)
        sharpe_ratio = (mean_rp - rf) / std_rp

        # Maximize Sharpe ratio by minimizing negative Sharpe ratio
        loss = -sharpe_ratio
        loss.backward()

        # Manual update of weights using gradient descent
        with torch.no_grad():
            w += learning_rate * w.grad

        # Optionally, print progress
        if i % 100 == 0:
            print(f"Iteration {i+1}/{iterations}")
            print(f"Normalized Weights: {w_normalized.detach().numpy()}")
            print(f"Sharpe Ratio: {sharpe_ratio.item()}")

    # Final optimized weights (normalized)
    optimized_weights = torch.nn.functional.softmax(w, dim=1)
    print("Optimized Weights (normalized):", optimized_weights.detach().numpy())

    # Sharpe ratio calculation is already based on daily data
    sharpe_ratio_daily = sharpe_ratio.item()
    print(f"Final Daily Sharpe Ratio: {sharpe_ratio_daily}")

    # Optionally annualize the Sharpe ratio for comparison with other tools
    sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(TRADING_DAYS)  # Annualize daily Sharpe ratio
    print(f"Annualized Sharpe Ratio: {sharpe_ratio_annualized}")
 