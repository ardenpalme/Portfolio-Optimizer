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

    bool get_price_series_since(const std::string &start_date) {
        std::string api_key, end_date;

        const std::chrono::time_point now{std::chrono::system_clock::now()};
        const std::chrono::year_month_day ymd{std::chrono::floor<std::chrono::days>(now)};
 
        end_date = std::format("{}",ymd);

        const char* api_key_ptr;
        if((api_key_ptr = std::getenv("POLYGON_API_KEY")) == NULL) {
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

        try {
            URL data_url(polygon_req);
            json_str_ptr = data_url.get_data();

        } catch(const CURLError &err) {
            std::cerr << err.what() << std::endl;
        }

        simdjson::padded_string padded_data(*json_str_ptr);
        auto error = parser.iterate(padded_data).get(json_doc);

        /*
        std::string_view status_str;
        error = json_doc["status"].get_string(status_str);
        if(status_str != "OK") {
            std::cerr << "Market Data request error " << status_str << std::endl;
            return false;
        }
        json_doc.rewind();
        */

        simdjson::ondemand::object json_obj = json_doc.get_object();
        for (auto field : json_obj) {
            simdjson::ondemand::raw_json_string key;
            error = field.key().get(key);
            if (error) return error;

            if (key == "results") {
                error = field.value().get<std::vector<double>>().get(price_series);
                if (error) return error;
            }
        }

        return true;
    }

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