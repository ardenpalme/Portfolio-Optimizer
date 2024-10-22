#include "auto_diff.hpp"

namespace AutoDiff {
    std::ostream& operator<<(std::ostream& os, Expression &expr) {
        if(expr.is_vector){
            os << "[vector] " << expr.value;
        }else{
            os << "[scalar] " << expr.scalar_value;
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, Variable &expr) {
        if(expr.is_vector){
            os << "[vector] " << expr.value;
            os << " [partial] " << expr.partial;
        }else{
            os << "[scalar] " << expr.scalar_value;
            os << " [partial] " << expr.partial_scalar;
        }
        return os;
    }
}