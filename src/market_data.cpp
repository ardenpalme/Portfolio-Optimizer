#include <Eigen/Dense>
#include <ctime>

#include "auto_diff.hpp"
#include "market_data.hpp"
#include "simdjson.h"
#include "libcurl.hpp"

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

    std::cout << polygon_req << std::endl;

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
        os << " ... " << *data_iter << "] (" << m_data.returns.size() << " total)" << std::endl;
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

std::ostream& operator<<(std::ostream &os, const Portfolio &port) {
    int M_max = 5;
    std::cout << "[returns]: " << port.returns.rows() << " x " << port.returns.cols() << std::endl;
    int j=0;
    for(auto row : port.returns.rowwise()) {
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
    for(auto row : port.covariance.rowwise()) {
        for(auto ele : row) {
            std::cout << ele << " ";
        }
        std::cout << std::endl;
    }

    return os;
}


bool Portfolio::optimize_sharpe(uint32_t num_epochs) { 
    AutoDiff::Variable w1(weights); 
    AutoDiff::LinearProd w2(&w1, mean);
    AutoDiff::VecT_Matrix_Vec w3(&w1, covariance);
    AutoDiff::Power w4(&w3, -0.5);
    AutoDiff::Multiply w5(&w4, &w2);

    Eigen::VectorXd seed = Eigen::VectorXd::Ones(weights.cols());

    w5.evaluate();
    std::cout << "f(w = [" << weights << "]) = " << w5.scalar_value << std::endl;

    w5.derive(seed);
    std::cout << "∂f/∂w = " << w1.partial << std::endl;
    return true; 
}