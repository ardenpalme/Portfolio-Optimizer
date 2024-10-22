#ifndef __BAYES_OPTIMIZER__
#define __BAYES_OPTIMIZER__

#include <Eigen/Dense>
#include <string>
#include <utility> 
#include <memory> 
#include <vector>   

using namespace std;
using namespace Eigen;

class KDE {
    string kernel_type;

public:
    KDE(string _kernel_type) : kernel_type(_kernel_type) {}

    double gaussian_kernel(double u);
    VectorXd evaluate(const VectorXd& data, double bandwidth, const VectorXd& points);
};

class OptObjective {
public:
    virtual double operator()(const VectorXd&, const MatrixXd& ) = 0;
};

class Omega : public OptObjective {
    KDE kernel_estimator;

public:
    Omega(const KDE& _kernel_estimator) : kernel_estimator{_kernel_estimator} {}

    double omega_ratio_kde(const VectorXd& returns, const VectorXd& kde_values);
    double operator()(const VectorXd& weights, const MatrixXd& asset_returns); 
};

class BayesOptimizer {
    std::unique_ptr<OptObjective> objective;
    
    double rbf_kernel(const VectorXd& x1, const VectorXd& x2, double length_scale = 1.0);
    
    // Acquisition function: Upper Confidence Bound (UCB)
    double ucb(const VectorXd& mu, const VectorXd& sigma, double beta = 2.0); 

    // Computes the covariance matrix for GP
    MatrixXd compute_covariance(const MatrixXd& X, double length_scale = 1.0);

    // GP posterior prediction
    pair<VectorXd, VectorXd> gp_predict(
        const MatrixXd& X_train, 
        const VectorXd& y_train,
        const VectorXd& x_new, 
        double noise = 1e-6, 
        double length_scale = 1.0); 


public:
    BayesOptimizer(std::unique_ptr<OptObjective> _objective) : objective(std::move(_objective)) {}

    // Bayesian Optimization using GP and UCB
    VectorXd optimize(const MatrixXd& asset_returns, int n_calls = 50);
};

#endif /* __BAYES_OPTIMIZER__ */