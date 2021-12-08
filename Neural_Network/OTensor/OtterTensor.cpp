//
//  OtterTensor.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/2.
//

#include "OtterTensor.hpp"

namespace otter {
// TensorNucleus main constructor
TensorNucleus::TensorNucleus(OtterMemory&& memory, const OtterType data_type, TensorOptionTODO) : memory_(std::move(memory)), data_type_(data_type), memory_offset_(0), numel_(0) {}

TensorNucleus::TensorNucleus(OtterMemory&& memory, const OtterType data_type) : TensorNucleus(std::forward<OtterMemory>(memory), data_type, TensorOptionTODO{}) {}

void TensorBase::print() const {
    if (this->defined()) {
        std::cerr << "[" << toString() << " " << sizes() << "]" << std::endl;
    } else {
        std::cerr << "[UndefinedTensor]" << std::endl;
    }
}

std::string TensorBase::toString() const {
    return ::toString(this->unsafeGetTensorNucleus()->scalar_type()) + "Type";
}


}   // namespace otter
