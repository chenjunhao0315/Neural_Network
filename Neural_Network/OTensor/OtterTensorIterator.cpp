//
//  OtterTensorIterator.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/9.
//

#include "OtterTensorIterator.hpp"

namespace otter {

InlineOtterTensorRef::InlineOtterTensorRef() {
    static_assert(alignof(OtterTensorRef) == alignof(TensorBase), "");
    static_assert(sizeof(OtterTensorRef) == sizeof(TensorBase), "");
    new (data_.data()) OtterTensorRef();
}

InlineOtterTensorRef::~InlineOtterTensorRef() {
    get()->~OtterTensorRef();
}

const Tensor& InlineOtterTensorRef::getTensor() const {
    return get()->getTensorRef();
}

static OtterTensorRef make_otr(const TensorBase &tensor) {
    if (tensor.defined()) {
        return OtterTensorRef(tensor);
    } else {
        return OtterTensorRef();
    }
}

void OtterOperandInfo::tensor(OtterMaybeOwned<TensorBase> &&tensor) {
    tensor_base_ = std::move(tensor);
    *tensor_storage_ = make_otr(*tensor_base_);
}

void OtterOperandInfo::exchange_tensor(OtterMaybeOwned<TensorBase> &&new_tensor) {
    assert(!original_tensor_base_->defined());
    original_tensor_base_ = std::exchange(tensor_base_, new_tensor);
    *original_tensor_storage_ = std::exchange(*tensor_storage_, make_otr(*tensor_base_));
}

void OtterOperandInfo::restore_original_tensor() {
    assert(original_tensor_base_->defined());
    tensor_base_ = std::move(original_tensor_base_);
    *tensor_storage_ = std::exchange(*original_tensor_storage_, OtterTensorRef{});
}

OtterTensorIteratorConfig& OtterTensorIteratorConfig::add_owned_output(const TensorBase& output) {
    tensors_.push_back(OtterMaybeOwned<TensorBase>::owned(in_place, output));
    outputs_num_++;
    return *this;
}

OtterTensorIteratorConfig& OtterTensorIteratorConfig::add_owned_input(const TensorBase& input) {
    tensors_.push_back(OtterMaybeOwned<TensorBase>::owned(in_place, input));
    inputs_num_++;
    return *this;
}

OtterTensorIteratorConfig& OtterTensorIteratorConfig::add_borrowed_output(const TensorBase& output) {
    tensors_.push_back(OtterMaybeOwned<TensorBase>::borrowed(output));
    outputs_num_++;
    return *this;
}

OtterTensorIteratorConfig& OtterTensorIteratorConfig::add_borrowed_input(const TensorBase& input) {
    tensors_.push_back(OtterMaybeOwned<TensorBase>::borrowed(input));
    inputs_num_++;
    return *this;
}

TensorIteratorBase::TensorIteratorBase() = default;

}   // end namespace otter
