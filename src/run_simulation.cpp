#include <iostream>

#include "auto_diff.hpp"
#include "market_data.hpp"

int main(int argc, char *argv[])
{
    using namespace std;

    std::string assets[] = {
        {"X:BTCUSD"},
        {"X:ETHUSD"}
    };

    std::vector<Market_Data> market_data_vec;
    for(auto asset : assets) {
        Market_Data m_data(asset);
        m_data.get_price_series_since("2023-04-10");
        cout << m_data << endl;
        market_data_vec.push_back(m_data);
    }

    Portfolio portfolio(market_data_vec);
    cout << portfolio << std::endl;

    portfolio.optimize_sharpe(1);

    return 0;
}