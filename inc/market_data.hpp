#ifndef __MARKET_DATA_HPP__
#define __MARKET_DATA_HPP__

#include <vector>
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
    Eigen::RowVectorXd mean; 
    Eigen::MatrixXd covariance;
    double sharpe_ratio;

public:
    Portfolio(std::vector<Market_Data> _assets);

    bool optimize_sharpe(uint32_t num_epochs);

    void print_matricies();
    friend std::ostream& operator<<(std::ostream &os, const Portfolio &port);
};

#endif /* __MARKET_DATA_HPP__ */