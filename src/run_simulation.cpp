#include <iostream>

#include "simdjson.h"
#include "market_data.hpp"

int main(int argc, char *argv[])
{
    using namespace std;

    Market_Data m_data("X:BTCUSD");
    m_data.get_price_series_since("2023-04-10");
    cout << m_data << endl;

    return 0;
}