#include "market_data.hpp"

bool Market_Data::get_price_series_since(const std::string &start_date)
{
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

    return true;
}

std::ostream& operator<<(std::ostream &os, const Market_Data &m_data) {
    if(m_data.price_series.size() == 0 ) return os;

    size_t count = 2;
    auto data_iter = m_data.price_series.begin();
    os << m_data.ticker << " price series [";
    while (data_iter != m_data.price_series.begin() + count) {
        os << *data_iter;
        if (data_iter + 1 != m_data.price_series.begin() + count) os << ", ";
        data_iter++;
    }
    data_iter = m_data.price_series.end() - 1;
    os << " ... " << *data_iter << "] (" << m_data.price_series.size() << " total)";
    return os;
}