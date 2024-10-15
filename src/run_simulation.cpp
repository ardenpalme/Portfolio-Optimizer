#include <iostream>

#include "auto_diff.hpp"
#include "market_data.hpp"

namespace AutoDiff {
    std::ostream& operator<<(std::ostream& os, Expression &expr) {
        if(expr.is_vector){
            os << "[vector] " << expr.value;
        }else{
            os << "[scalar] " << expr.scalar_value;
        }
        return os;
    }
}

int main(int argc, char *argv[])
{
    using namespace std;

    std::string assets[] = {
        {"X:BTCUSD"},
        {"X:ETHUSD"},
        {"X:SOLUSD"}
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

    portfolio.optimize_sharpe(100);

    return 0;
}