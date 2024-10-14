#include <iostream>

#include "simdjson.h"
#include "market_data.hpp"

std::ostream& operator<<(std::ostream &os, const Market_Data &m_data) {
    size_t count = 2;
    auto data_iter = m_data.price_series.begin();
    os << "[";
    while (data_iter != m_data.price_series.begin() + count) {
        os << *data_iter;
        if (data_iter + 1 != m_data.price_series.begin() + count) os << ", ";
        data_iter++;
    }
    data_iter = m_data.price_series.end() - 1;
    os << " ... " << *data_iter << "] (" << m_data.price_series.size() << " total)";
    return os;
}

int main(int argc, char *argv[])
{
    using namespace std;

    Market_Data m_data("X:BTCUSD");
    //m_data.get_price_series_since("2023-04-10");
    cout << m_data << endl;

    return 0;
}