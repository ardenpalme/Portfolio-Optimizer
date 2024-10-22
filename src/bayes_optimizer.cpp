#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <Eigen/Dense>

#include "bayes_optimizer.hpp"

double Omega::omega_ratio_kde(const VectorXd& returns, const VectorXd& kde_values) {
    double threshold = 0.0;
    double gain = kde_values(returns.array() > threshold).sum();
    double loss = kde_values(returns.array() <= threshold).sum();
    return gain / loss;
}

// Objective function (for the GP)
double Omega::operator()(const VectorXd& weights, const MatrixXd& asset_returns) {
    VectorXd w = weights / weights.sum();
    VectorXd rp = asset_returns.transpose() * w;

    double variance = (rp.array() - rp.mean()).square().sum() / (rp.size() - 1);  
    double standard_dev =  std::sqrt(variance);

    VectorXd kde_values = kernel_estimator.evaluate(rp, 1.06 * standard_dev * pow(rp.size(), -0.2), rp);
    double omega = omega_ratio_kde(rp, kde_values);

    return -omega;
}

VectorXd KDE::evaluate(const VectorXd& data, double bandwidth, const VectorXd& points)
{
    int n = data.size();
    int m = points.size();
    VectorXd kde_values(m);

    for (int j = 0; j < m; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            double u = (points(j) - data(i)) / bandwidth;
            sum += gaussian_kernel(u);
        }
        kde_values(j) = sum / (n * bandwidth);
    }

    return kde_values;
}

double KDE::gaussian_kernel(double u)
{
    return exp(-0.5 * u * u) / sqrt(2.0 * M_PI);
}

double BayesOptimizer::rbf_kernel(const VectorXd& x1, const VectorXd& x2, double length_scale)
{
    return exp(-(x1 - x2).squaredNorm() / (2 * length_scale * length_scale));
}

MatrixXd BayesOptimizer::compute_covariance(const MatrixXd& X, double length_scale) 
{
    int n = X.rows();
    MatrixXd K(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            K(i, j) = rbf_kernel(X.row(i), X.row(j), length_scale);
            K(j, i) = K(i, j);
        }
    }
    return K;
}

double BayesOptimizer::ucb(const VectorXd& mu, const VectorXd& sigma, double beta) 
{
    return mu(0) + beta * sigma(0);
} 

pair<VectorXd, VectorXd> BayesOptimizer::gp_predict(
    const MatrixXd& X_train, 
    const VectorXd& y_train,
    const VectorXd& x_new, 
    double noise,
    double length_scale)
{
    MatrixXd K = compute_covariance(X_train, length_scale) + noise * MatrixXd::Identity(X_train.rows(), X_train.rows());
    VectorXd k_star(X_train.rows());
    for (int i = 0; i < X_train.rows(); ++i) {
        k_star(i) = rbf_kernel(X_train.row(i), x_new, length_scale);
    }

    LLT<MatrixXd> llt(K);
    VectorXd alpha = llt.solve(y_train);

    double mu = k_star.dot(alpha);
    double var = rbf_kernel(x_new, x_new, length_scale) - k_star.transpose() * llt.solve(k_star);

    return {VectorXd::Constant(1, mu), VectorXd::Constant(1, sqrt(var))};
}

VectorXd BayesOptimizer::optimize(const MatrixXd& asset_returns, int n_calls) {
    vector<VectorXd> X_train;
    VectorXd y_train(n_calls);

    int num_assets = asset_returns.rows(); 

    // Initialize with random points
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < num_assets; ++i) {
        VectorXd weights(num_assets);
        for (int j = 0; j < num_assets; ++j) {
            weights(j) = dis(gen);
        }
        X_train.push_back(weights);
        y_train(i) = (*objective)(weights, asset_returns);
    }

    VectorXd best_weights = X_train[0];
    double best_value = y_train(0);

    for (int call = num_assets; call < n_calls; ++call) {
        // GP Prediction and UCB Acquisition
        VectorXd mu, sigma;
        double max_ucb = -INFINITY;
        VectorXd next_point = VectorXd::Zero(num_assets);

        for (const auto& x : X_train) {
            tie(mu, sigma) = gp_predict(Map<MatrixXd>(X_train[0].data(), X_train.size(), num_assets), y_train, x);
            double current_ucb = ucb(mu, sigma);

            if (current_ucb > max_ucb) {
                max_ucb = current_ucb;
                next_point = x;
            }
        }

        // Evaluate the objective at the new point
        double new_value = (*objective)(next_point, asset_returns);

        // Update the training set
        X_train.push_back(next_point);
        y_train.conservativeResize(call + 1);
        y_train(call) = new_value;

        // Check for improvement
        if (new_value < best_value) {
            best_value = new_value;
            best_weights = next_point;
        }
    }

    return best_weights / best_weights.sum();  // Normalize the weights
}
