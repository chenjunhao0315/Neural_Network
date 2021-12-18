//
//  OtterTensorIterator.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/9.
//

#ifndef OtterTensorIterator_hpp
#define OtterTensorIterator_hpp

#include <stdio.h>
#include <array>

#include "OtterArrayRef.hpp"
#include "OtterSmallVector.hpp"
#include "OtterMaybeOwned.hpp"
#include "OtterTensor.hpp"
#include "OtterTensorOption.hpp"

namespace otter {

class OtterTensorRef {
public:
    OtterTensorRef() = default;
    
    ~OtterTensorRef() {
        ref_.unsafeReleaseTensorNucleus();
    }
    
    OtterTensorRef(const TensorBase& src) : ref_(Tensor::unsafe_borrow_t{}, src) {
        assert(src.defined());
    }
    
    OtterTensorRef(const OtterTensorRef& rhs) : ref_(Tensor::unsafe_borrow_t{}, rhs.ref_) {}
    
    OtterTensorRef& operator=(OtterTensorRef rhs) {
        std::swap(ref_, rhs.ref_);
        return *this;
    }
    
    bool has_value() const {
        return ref_.defined();
    }
    
    const Tensor& getTensorRef() const & {
        return ref_;
    }
    
    const Tensor& operator*() const & {
        return ref_;
    }
    
    const Tensor* operator->() const & {
        return &ref_;
    }
    
    operator bool() const {
        return ref_.defined();
    }
    
private:
    Tensor ref_;
};

class InlineOtterTensorRef {
    alignas(alignof(TensorBase)) std::array<char, sizeof(TensorBase)> data_;
public:
    InlineOtterTensorRef();
    ~InlineOtterTensorRef();
    
    OtterTensorRef* get() {
        return reinterpret_cast<OtterTensorRef*>(data_.data());
    }
    const OtterTensorRef* get() const {
        return reinterpret_cast<const OtterTensorRef*>(data_.data());
    }
    
    OtterTensorRef& operator*() { return *get(); }
    const OtterTensorRef& operator*() const { return *get(); }
    OtterTensorRef* operator->() { return get(); }
    const OtterTensorRef* operator->() const { return get(); }
    
    const Tensor& getTensor() const;
};

struct OtterOperandInfo {
    using StrideVector = SmallVector<int64_t, 6>;
    
    OtterOperandInfo() = default;
    
    ~OtterOperandInfo() = default;
    
    explicit OtterOperandInfo(OtterMaybeOwned<TensorBase>&& t) {
        if (t->defined()) {
            //            device = t->device();
            //            target_dtype = t->scalar_type();
            //            current_dtype = target_dtype;
        }
        //        tensor(std::move(t));
        //        validate();
    }
    
    void validate() {
        assert(!tensor_base_->defined());
    }
    
    bool is_type_defined() const { return target_dtype != ScalarType::Undefined; }
    
    TensorOption options() const {
        return TensorOption(target_dtype).device(device);
    }
    
    const Tensor& tensor() const {
        return tensor_storage_.getTensor();
    }
    const TensorBase& tensor_base() const {
        return *tensor_base_;
    }
    void tensor(OtterMaybeOwned<TensorBase> &&tensor);
    
    const Tensor& original_tensor() const {
        return original_tensor_storage_.getTensor();
    }
    const TensorBase& original_tensor_base() const {
        return *original_tensor_base_;
    }
    
    void exchange_tensor(OtterMaybeOwned<TensorBase> &&new_tensor);

    void restore_original_tensor();
    
    StrideVector stride_bytes;
    ScalarType target_dtype = ScalarType::Undefined;
    ScalarType current_dtype = ScalarType::Undefined;
    Device device = Device::CPU;
    
private:
    OtterMaybeOwned<TensorBase> tensor_base_;
    OtterMaybeOwned<TensorBase> original_tensor_base_ = OtterMaybeOwned<TensorBase>::owned(otter::in_place);
    
    InlineOtterTensorRef tensor_storage_;
    InlineOtterTensorRef original_tensor_storage_;
};

struct MetaBase {
    virtual void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOption options) = 0;
    virtual const Tensor& maybe_get_output(int64_t output_idx) = 0;
    void set_output(IntArrayRef sizes, TensorOption options) {
        set_output(0, sizes, {}, options);
    }
    void set_output(int64_t output_idx, IntArrayRef sizes, TensorOption options) {
        set_output(output_idx, sizes, {}, options);
    }
    // Returns a reference to an undefined tensor if there is no presupplied
    // output
    const Tensor& maybe_get_output() { return maybe_get_output(0); }
    virtual ~MetaBase() {}
};

class OtterTensorIteratorConfig;

constexpr size_t kDimVectorStaticSize = 5;
using DimVector = SmallVector<int64_t, kDimVectorStaticSize>;

struct TensorIteratorBase : public MetaBase {
    using StrideVector = SmallVector<int64_t, 6>;
    
    TensorIteratorBase();
    void build(OtterTensorIteratorConfig&);
    
    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOption options) {}
    const Tensor& maybe_get_output(int64_t output_idx) { return Tensor(); }
    
protected:
    DimVector shape_;
    DimVector perm_;
    DimVector view_offsets_;
    
    SmallVector<OtterOperandInfo, 4> operands_;
    
    ScalarType common_dtype_ = ScalarType::Undefined;
    
    Device common_device_ = Device::CPU;
    
    int num_outputs_ = 0;
    
    bool all_ops_same_shape_ = false;
    bool accumulate_ = false;
    bool final_output_ = true;
};

struct TensorIterator : public TensorIteratorBase {
    TensorIterator() : TensorIteratorBase() {}
    //    TensorIterator(const TensorIteratorBase& iter) : TensorIteratorBase(iter) {}
};

class OtterTensorIteratorConfig {
public:
    OtterTensorIteratorConfig() {}
    
    OtterTensorIteratorConfig(const OtterTensorIteratorConfig&) = delete;
    OtterTensorIteratorConfig& operator=(const OtterTensorIteratorConfig&) = delete;
    
    OtterTensorIteratorConfig& add_output(const TensorBase& output) {
        return add_borrowed_output(output);
    }
    OtterTensorIteratorConfig& add_input(const TensorBase& input) {
        return add_borrowed_input(input);
    }
    
    // Can't build from temporaries
    OtterTensorIteratorConfig& add_output(TensorBase&& output) = delete;
    OtterTensorIteratorConfig& add_input(TensorBase&& input) = delete;
    
    // Increase reference counter
    OtterTensorIteratorConfig& add_owned_output(const TensorBase& output);
    OtterTensorIteratorConfig& add_owned_input(const TensorBase& input);
    
    // Can't build from temporaries
    OtterTensorIteratorConfig& add_borrowed_output(TensorBase&& output) = delete;
    OtterTensorIteratorConfig& add_borrowed_input(TensorBase&& input) = delete;
    
    // Ensure the lifetime of Tensor is longer than OtterTensorIteratorConfig
    // output -> input
    OtterTensorIteratorConfig& add_borrowed_output(const TensorBase& output);
    OtterTensorIteratorConfig& add_borrowed_input(const TensorBase& input);
    
    OtterTensorIteratorConfig& set_check_mem_overlap(bool check_mem_overlap) {
        check_mem_overlap_ = check_mem_overlap;
        return *this;
    }
    
    OtterTensorIteratorConfig& set_resize_outputs(bool resize_output) {
        resize_outputs_ = resize_output;
        return *this;
    }
    
    OtterTensorIteratorConfig& set_check_all_same_dtype(bool check_all_same_dtype) {
        check_all_same_dtype_ = check_all_same_dtype;
        return *this;
    }
    
    TensorIterator build() {
        TensorIterator iter;
        //        iter.build(*this);
        return iter;
    }
    
private:
    SmallVector<OtterMaybeOwned<TensorBase>, 4> tensors_;
    int inputs_num_ = 0;
    int outputs_num_ = 0;
    
    bool check_mem_overlap_ = true;
    bool resize_outputs_ = true;
    bool check_all_same_dtype_ = true;
};

}   // end namespace otter

#endif /* OtterTensorIterator_hpp */
