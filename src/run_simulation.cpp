#include <iostream>

#include "auto_diff.hpp"
#include "market_data.hpp"

int main(int argc, char *argv[])
{
    /*
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
    */

    using namespace AutoDiff;
    // f(x1,x2) = x1 * (x2  ** (-1/2))
    Variable x1(1), x2(25);
    Power p1(&x2, -0.5);
    Multiply z(&x1, &p1);

    z.evaluate();
    std::cout << "f(x1 = " << x1.value << ", x2 = " << x2.value << ") = " << z.value << std::endl;

    z.derive(1);
    std::cout << "∂f/∂x1 = " << x1.partial << std::endl
              << "∂f/∂x2 = " << x2.partial << std::endl;

    return 0;
}