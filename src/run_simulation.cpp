#include <iostream>

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
        m_data.get_price_series_since("2000-01-01");
        cout << m_data << endl;
        market_data_vec.push_back(m_data);
    }

    Portfolio portfolio(market_data_vec);
    portfolio.print_matricies();

    portfolio.optimize_sharpe(10);
    cout << portfolio << std::endl;

    portfolio.optimize_omega();
    cout << portfolio << std::endl;

    return 0;
}