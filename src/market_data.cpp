#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>

#include "auto_diff.hpp"
#include "market_data.hpp"
#include "simdjson.h"
#include "libcurl.hpp"
#include "bayes_optimizer.hpp"

#define TRADING_DAYS 365

bool Market_Data::get_price_series_since(const std::string &start_date)
{
    std::vector<double> price_series;
    std::string api_key, end_date;

    const std::chrono::time_point now{std::chrono::system_clock::now()};
    const std::chrono::year_month_day ymd{std::chrono::floor<std::chrono::days>(now)};

    end_date = std::format("{}", ymd);

    const char *api_key_ptr;
    if ((api_key_ptr = std::getenv("POLYGON_API_KEY")) == NULL)
    {
        std::cerr << "Invalid API Key for polygon.io market data service" << std::endl;
        return false;
    }
    api_key = std::string(api_key_ptr);

    std::string polygon_req =
        std::format("https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/{}/{}?adjusted=true&sort=asc&apiKey={}",
                    ticker,
                    start_date,
                    end_date,
                    api_key);

#ifdef DEBUG_CURL_JSON
    std::cout << polygon_req << std::endl;
#endif 

    simdjson::ondemand::document json_doc;
    simdjson::ondemand::parser parser;
    std::shared_ptr<std::string> json_str_ptr;

    try
    {
        URL data_url(polygon_req);
        json_str_ptr = data_url.get_data();
    }
    catch (const CURLError &err)
    {
        std::cerr << err.what() << std::endl;
    }

    simdjson::padded_string padded_data(*json_str_ptr);
    auto error = parser.iterate(padded_data).get(json_doc);

    std::string_view status_str;
    error = json_doc["status"].get_string(status_str);
    if (status_str != "OK")
    {
        std::cerr << "Market Data request error " << status_str << std::endl;
        return false;
    }
    json_doc.rewind();

    // Extract closing price from polygon market data JSON
    simdjson::ondemand::object json_obj = json_doc.get_object();
    for (auto field : json_obj)
    {
        simdjson::ondemand::raw_json_string key;
        error = field.key().get(key);
        if (error)
            return error;

        if (key == "results")
        {
            error = field.value().get<std::vector<double>>().get(price_series);
            if (error)
                return error;
        }
    }

    auto iter = price_series.begin();
    auto prev_iter = price_series.end();

    while(iter != price_series.end()) {
        if(prev_iter == price_series.end()) {
            prev_iter = iter;
            iter = std::next(iter);
            continue;
        }

        double percent_diff = (*iter - *prev_iter) / *prev_iter; 
        returns.push_back(percent_diff);

        prev_iter = iter;
        iter = std::next(iter);
    }

    return true;
}

std::ostream& operator<<(std::ostream &os, const Market_Data &m_data) {
    if(m_data.returns.size() > 0 ) {
        size_t count = 2;
        auto data_iter = m_data.returns.begin();
        os << m_data.ticker << " returns [";
        while (data_iter != m_data.returns.begin() + count) {
            os << *data_iter;
            if (data_iter + 1 != m_data.returns.begin() + count) os << ", ";
            data_iter++;
        }
        data_iter = m_data.returns.end() - 1;
        os << " ... " << *data_iter << "] (" << m_data.returns.size() << " total)";
    }

    return os;
}

namespace simdjson {

    template <>
    simdjson_inline simdjson_result<std::vector<double>>
    simdjson::ondemand::value::get() noexcept
    {
        std::vector<double> vec;

        ondemand::array arr;
        auto error = get_array().get(arr);
        if (error) return error;

        for (auto ele : arr) {
            ondemand::object obj; 
            error = ele.get_object().get(obj);
            if (error) return error;

            for (auto field : obj) {
                double close_price;
                simdjson::ondemand::raw_json_string key;
                error = field.key().get(key);
                if (error) return error;

                if (key == "c") {
                    error = field.value().get_double().get(close_price);
                    vec.push_back(close_price);
                    if (error) return error;
                }
            }
        }
        return vec;
    }
};

void Portfolio::print_matricies()
{
    int M_max = 5;
    std::cout << "[returns]: " << returns.rows() << " x " << returns.cols() << std::endl;
    int j=0;
    for(auto row : returns.rowwise()) {
        int i=0;
        for(auto ele : row) {
            std::cout << ele << " ";
            if(++i > M_max) break;
        }
        std::cout << std::endl;
        if(++j > M_max) break;
    }
    std::cout << std::endl;

    std::cout << "[covariance]: " << std::endl;
    for(auto row : covariance.rowwise()) {
        for(auto ele : row) {
            std::cout << ele << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::ostream& operator<<(std::ostream &os, const Portfolio &port) {
    os << "Allocations: "  
       << "(Annualized Sharpe Ratio (ex-post) = " << port.sharpe_ratio << ")" << std::endl;
    int idx = 0;
    for(auto asset : port.assets) {
        os << "[" << asset.ticker << " " 
           << port.weights[idx++] << "]" << std::endl;
    }
    os << std::endl;

    return os;
}

Portfolio::Portfolio(std::vector<Market_Data> _assets) : assets{_assets} {
    size_t num_assets = assets.size();
    size_t data_len = assets.at(0).returns.size();
    for(auto asset : assets) {
        size_t len = asset.returns.size();
        if(len < data_len) data_len = len;
    }
    
    //std::cout << "N = " << num_assets << " M = " << data_len << std::endl;

    Eigen::MatrixXd returns_NM(num_assets, data_len);
    int row_idx = 0;
    std::for_each(assets.begin(), assets.end(), [&](const auto &m_data){
        std::vector<double> returns_vec = m_data.returns;
        Eigen::Map<Eigen::RowVectorXd> row_vec(returns_vec.data(), data_len);
        returns_NM.row(row_idx++) = row_vec;
    });

    returns = std::move(returns_NM);

    Eigen::VectorXd mu = returns.rowwise().mean();  
    Eigen::MatrixXd centered = returns.colwise() - mu; // (X- X_bar)

    int M = returns.cols();
    covariance = (centered * centered.transpose()) / (M - 1);

    mean = std::move(mu.transpose());

    std::srand(std::time(0)); 
    Eigen::RowVectorXd init_weights = Eigen::RowVectorXd::Random(num_assets).cwiseAbs();
    init_weights /= init_weights.sum();
    weights = std::move(init_weights);
}

bool Portfolio::optimize_sharpe(uint32_t num_epochs) { 
    double sharpe;
    const double learning_rate = 0.01;
    const double tolerance = 1e-9;

    std::cout << "Sharpe Ratio Optimization" << std::endl;
    for(int i=0; i<num_epochs; i++) {

        // Reinitialize the computation graph 
        AutoDiff::Variable w1(weights); 
        AutoDiff::LinProd w2(&w1, mean);            // Expected returns: w^T * mean
        AutoDiff::QuadProd w3(&w1, covariance);     // Portfolio variance: w^T * Cov * w
        AutoDiff::Pow w4(&w3, -0.5);                // Volatility: (w^T * Cov * w)^(-0.5)
        AutoDiff::ElemProd w5(&w4, &w2);            // Sharpe ratio: (w^T * mean) / sqrt(w^T * Cov * w)

        Eigen::RowVectorXd seed = Eigen::RowVectorXd::Ones(weights.cols());

        w5.evaluate();
        sharpe = w5.scalar_value;
        //std::cout << "S(w = [" << weights << "]) = " << w5.scalar_value << std::endl;

        w5.derive(seed);
        //std::cout << "∂S/∂w = " << w1.partial << std::endl;
        assert(std::abs(weights.array().sum() - 1.0) < tolerance);

        weights.array() += (learning_rate * w1.partial.array());
        weights.array() /= weights.array().sum();
    }

    sharpe_ratio = sharpe * std::sqrt(TRADING_DAYS);

    return true; 
}

void Portfolio::optimize_omega(uint32_t num_epochs) { 
    KDE gauss_kernel("gaussian");
    unique_ptr<OptObjective> omega = make_unique<Omega>(gauss_kernel);
    BayesOptimizer bayes_opt(std::move(omega));

    VectorXd new_weights = bayes_opt.optimize(returns, num_epochs);
    
    std::cout << "[ "; 
    for (auto alloc : new_weights) {
        std::cout << alloc << " ";
    }
    std::cout << "]" << std::endl; 
}
