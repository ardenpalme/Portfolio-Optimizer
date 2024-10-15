#ifndef __MARKET_DATA_HPP__
#define __MARKET_DATA_HPP__

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Dense>


struct Market_Data {
    std::vector<double> returns;
    std::string ticker;

    Market_Data(const std::string &_ticker) : ticker{_ticker} {}

    bool get_price_series_since(const std::string &start_date);

    friend std::ostream& operator<<(std::ostream &os, const Market_Data &m_data);
};

class Portfolio {
    std::vector<Market_Data> assets;
    Eigen::RowVectorXd weights;
    Eigen::MatrixXd returns;
    Eigen::VectorXd mean; 
    Eigen::MatrixXd covariance;

public:
    Portfolio(std::vector<Market_Data> _assets) : assets{_assets} {
        size_t num_assets = assets.size();
        size_t data_len = assets.at(0).returns.size();
        std::cout << "N = " << num_assets << " M = " << data_len << std::endl;

        Eigen::MatrixXd ret(num_assets, data_len);
        int row_idx = 0;
        std::for_each(assets.begin(), assets.end(), [&](const auto &m_data){
            std::vector<double> returns_vec = m_data.returns;
            Eigen::Map<Eigen::RowVectorXd> row_vec(returns_vec.data(), data_len);
            ret.row(row_idx++) = row_vec;
        });

        returns = std::move(ret);

        mean = returns.rowwise().mean();  
        Eigen::MatrixXd centered = returns.colwise() - mean; // (X- X_bar)

        int M = returns.cols();
        covariance = (centered * centered.transpose()) / (M - 1);

        std::srand(std::time(0)); 
        Eigen::RowVectorXd init_weights = Eigen::RowVectorXd::Random(2).cwiseAbs();
        init_weights /= init_weights.sum();
        weights = std::move(init_weights);
    }

    bool optimize_sharpe(uint32_t num_epochs);

    friend std::ostream& operator<<(std::ostream &os, const Portfolio &port);
};

#endif /* __MARKET_DATA_HPP__ */