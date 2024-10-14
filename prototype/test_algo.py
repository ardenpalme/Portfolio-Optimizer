import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
import requests
import numpy as np
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
    asset_1 = MarketData("BTCUSD", 365)
    asset_2 = MarketData("ETHUSD", 365)

    asset_1.get_data()
    asset_2.get_data()

    asset_1_2_cov = np.cov(asset_1.price_return, asset_2.price_return)
    print(asset_1_2_cov, "")

    '''
    fig, axes = plt.subplots(1,2)
    asset_1.plot_returns(axes[0])
    asset_2.plot_returns(axes[1])
    plt.show()
    '''

    asset_returns = np.array([asset_1.price_return, asset_2.price_return])
    price_returns = torch.tensor(asset_returns, dtype=torch.float32, requires_grad=True)
    w = torch.tensor(np.random.rand(1,2), dtype=torch.float32, requires_grad=True)  # Initial portfolio weights

    # Risk-free rate
    rf = 0.0

    # Learning rate for SGD
    learning_rate = 0.01

    # Number of iterations (stopping criterion)
    iterations = 1000

    # Perform SGD for the specified number of iterations
    for i in range(iterations):
        
        # Zero the gradient from the previous iteration
        w.grad = None

        rp = torch.matmul(w, price_returns)

        # Compute the mean return and standard deviation
        mean_rp = torch.mean(rp)
        var_rp = torch.var(rp, unbiased=False)
        std_rp = torch.sqrt(var_rp)

        # Compute the Sharpe ratio
        sharpe_ratio = (mean_rp - rf) / std_rp

        # Compute the gradient of the Sharpe ratio with respect to weights
        sharpe_ratio.backward()

        # Gradient of the weights
        grad_w = w.grad

        # Update the weights using the SGD update rule
        with torch.no_grad():
            w += learning_rate * grad_w

        # Optionally, print the progress
        if i % 100 == 0:
            print(f"Iteration {i+1}/{iterations}")
            print(f"Weights: {w.detach().numpy()}")
            print(f"Sharpe Ratio: {sharpe_ratio.item()}")
            print(f"Gradient: {grad_w.numpy()}")

    # Final optimized weights
    print("Optimized Weights:", w.detach().numpy())