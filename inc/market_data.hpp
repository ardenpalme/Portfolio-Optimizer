#ifndef __MARKET_DATA_HPP__
#define __MARKET_DATA_HPP__

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstdio>

#include "simdjson.h"
#include "libcurl.hpp"

struct Market_Data
{
    std::vector<double> price_series;
    std::vector<double> returns;
    std::string ticker;

    Market_Data(const std::string &_ticker) : ticker{_ticker} {}

    bool get_price_series_since(const std::string &start_date);

    friend std::ostream& operator<<(std::ostream &os, const Market_Data &m_data);
};

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

#endif /* __MARKET_DATA_HPP__ */