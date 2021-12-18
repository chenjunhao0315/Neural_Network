//
//  OtterScalarType.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/10.
//

#include "OtterScalarType.hpp"

std::string toString(ScalarType type) {
    switch(type) {
#define DEFINE_STR(_1, n) case ScalarType::n: return std::string(#n); break;
        OTTER_ALL_SCALAR_TYPES(DEFINE_STR)
#undef DEFINE_STR
        default:
            return std::string("Undefined");
    }
}
