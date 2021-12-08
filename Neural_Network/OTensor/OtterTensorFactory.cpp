//
//  OtterTensorFactory.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/9.
//

#include "OtterTensorFactory.hpp"

namespace otter {

otter::Tensor empty_cpu(IntArrayRef size, ScalarType dtype) {
    return empty_raw(size, GetAllocator(Device::CPU), OtterType::fromScalarType(dtype), Device::CPU);
}

otter::Tensor empty_raw(IntArrayRef size, Allocator* allocator, OtterType dtype, Device device) {
    int64_t nelements = multiply_integers(size);
    int64_t size_bytes = nelements * dtype.itemsize();
    
    OtterMemory memory = make_otterptr<MemoryNucleus>(size_bytes, allocator);
    Tensor tensor = otter::make_tensor<otter::TensorNucleus>(std::move(memory), dtype);
    tensor.unsafeGetTensorNucleus()->set_sizes_contiguous(size);
    
    return tensor;
}

}   // namespace otter
