//
//  OtterScalar.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/10.
//

#include "OtterScalar.hpp"

namespace otter {

Scalar Scalar::operator-() const {
    if (isFloatingPoint()) {
        return Scalar(-v.d);
    }
    return Scalar(-v.i);
}

Scalar Scalar::log() const {
    if (isFloatingPoint()) {
        return Scalar(std::log(v.d));
    }
    return Scalar(std::log(v.i));
}

}   // namespace otter
