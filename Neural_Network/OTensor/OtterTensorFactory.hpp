//
//  OtterTensorFactory.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/9.
//

#ifndef OtterTensorFactory_hpp
#define OtterTensorFactory_hpp

#include <stdio.h>

#include "OtterTensor.hpp"
#include "OtterAccumulator.hpp"

namespace otter {

otter::Tensor empty_cpu(IntArrayRef size, ScalarType dtype);

otter::Tensor empty_raw(IntArrayRef size, Allocator* allocator, OtterType dtype, Device device);

}   // namespace otter

#endif /* OtterTensorFactory_hpp */
