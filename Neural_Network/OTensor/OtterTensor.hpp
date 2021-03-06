//
//  OtterTensor.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/2.
//

#ifndef OtterTensor_hpp
#define OtterTensor_hpp

#include <stdio.h>

#include "OtterMemory.hpp"
#include "OtterDType.hpp"
#include "OtterPerspectiveView.hpp"
#include "OtterMaybeOwned.hpp"

#define NOT_IMPLEMENTED fprintf(stderr, "NOT_IMPLEMENTED!")

struct TensorOptionTODO {};

namespace otter {
class TensorNucleus : public OtterPtr_quantum {
public:
    TensorNucleus() = delete;
    TensorNucleus(const TensorNucleus&) = delete;
    TensorNucleus& operator=(const TensorNucleus&) = delete;
    TensorNucleus(TensorNucleus&&) = delete;
    TensorNucleus& operator=(TensorNucleus&&) = delete;
    
    TensorNucleus(OtterMemory&& memory, const OtterType data_type, TensorOptionTODO);
    
    TensorNucleus(OtterMemory&& memory, const OtterType data_type);
    
    void release_resources() override {
        memory_ = {};
    }
    
    int64_t dim() const {
        return perspective_view_.size();
    }
    
    int64_t size(size_t idx) const {
        // Maybe do some check
        return perspective_view_.size_at(idx);
    }
    
    IntArrayRef sizes() const {
        return perspective_view_.sizes_arrayref();
    }
    
    int64_t stride(size_t idx) const {
        // Maybe do some check
        return perspective_view_.stride_at(idx);
    }
    
    IntArrayRef strides() const {
        return perspective_view_.strides_arrayref();
    }
    
    void set_sizes_and_strides(IntArrayRef newSizes, IntArrayRef newStrides) {
        if (newSizes.size() != newStrides.size())
            fprintf(stderr, "[TensorNucleus] Dimensionality of sizes (%zu) must match dimensionality of strides(%zu)!\n", newSizes.size(), newStrides.size());
        const int64_t new_dim = newSizes.size();
        
        perspective_view_.set_sizes(newSizes);
        if (new_dim > 0) {
            for (size_t dim = new_dim; dim--; ) {
                if (newStrides[dim] >= 0) {
                    perspective_view_.stride_at(dim) = newStrides[dim];
                } else {
                    // XXX: This behavior is surprising and may need to be removed to
                    // support negative strides. Some pytorch functions rely on it:
                    // for example, torch.cat (run TestTorch.test_cat_empty).
                    if (dim == new_dim - 1) {
                        perspective_view_.stride_at(dim) = 1;
                    } else {
                        // Keep stride monotonically increasing to match NumPy.
                        perspective_view_.stride_at(dim) = (std::max<int64_t>(perspective_view_.size_at(dim + 1), 1)) * perspective_view_.stride_at(dim + 1);
                    }
                }
            }
        }
        this->update_numel();
    }
    
    int64_t numel() const {
        return numel_;
    }
    
    int64_t compute_numel() const {
        int64_t n = 1;
        for (auto s : this->sizes()) {
            n *= s;
        }
        return n;
    }
    
    void update_numel() {
        numel_ = compute_numel();
    }
    
    bool is_empty() const {
        return numel_ == 0;
    }
    
    bool memory_initialized() {
        return memory_.data() || numel_ == 0;
    }
    
    bool dtype_initialized() {
        return data_type_ != OtterType();
    }
    
    template <typename T>
    inline T* data() const {
        return data_ptr_nucleus<T>();
    }
    
    template <typename T>
    inline T* data_ptr_nucleus() const {
        return memory_.unsafe_data<T>() + memory_offset_;
    }
    
    inline void* data() const {
        if (this->is_empty()) {
            return nullptr;
        }
        return static_cast<void*>(static_cast<char*>(memory_.data()) + data_type_.itemsize() * memory_offset_);
    }
    
    template<typename T>
    inline T* mutable_data() {
        if (memory_initialized() && data_type_.Match<T>()) {
            return static_cast<T*>(memory_.data()) + memory_offset_;
        }
        return static_cast<T*>(raw_mutable_data(OtterType::Make<T>()));
    }
    
    inline void* raw_mutable_data(const OtterType type) {
        if (data_type_ == type && memory_initialized()) {
            return static_cast<void*>(static_cast<char*>(memory_.data()) + memory_offset_ * type.itemsize());
        } else {
            memory_offset_ = 0;
            data_type_ = type;
            
            const Allocator* allocator = memory_.allocator();
            if (allocator == nullptr) {
                allocator = GetAllocator(memory_.device());
            }
            if (type.placementNew()) {
                NOT_IMPLEMENTED;
            } else {
                memory_.set_data_ptr_noswap(allocator->allocate(numel_ * type.itemsize()));
            }
            memory_.set_nbytes(numel_ * type.itemsize());
            return memory_.data();
        }
    }
    
    OtterType dtype() const {
        return data_type_;
    }
    
    ScalarType scalar_type() {
        return data_type_.toScalarType();
    }
    
    size_t itemsize() const {
        return data_type_.itemsize();
    }
    
    int64_t memory_offset() const {
        return memory_offset_;
    }
    
    bool has_memory() const {
        return memory_;
    }
    
    const OtterMemory& memory() const {
        return memory_;
    }
    
    inline void FreeMemory() {
        memory_ = OtterMemory::create_empty(memory_.device());
        memory_offset_ = 0;
    }
    
    void set_sizes_contiguous(IntArrayRef newSizes) {
        perspective_view_.set_sizes(newSizes);
        this->update_numel();
        this->restride();
    }
    
    void restride() {
        const int64_t dim_ = dim();
        //        perspective_view_.resize(dim_);
        if (dim_ > 0) {
            const int64_t last_idx = dim_ - 1;
            perspective_view_.stride_at(last_idx) = 1;
            for (int64_t i = last_idx; i--; ) {
                perspective_view_.stride_at(i) = perspective_view_.stride_at(i + 1) * std::max<int64_t>(perspective_view_.size_at(i + 1), 1);
            }
        }
    }
    
    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool SetDimsTemplate(OtterArrayRef<T> src) {
        int64_t old_numel = numel_;
        perspective_view_.resize(src.size());
        int64_t new_numel = 1;
        for (size_t i = 0; i < src.size(); ++i) {
            new_numel *= src[i];
            perspective_view_.size_at(i) = src[i];
        }
        numel_ = new_numel;
        this->restride();
        return numel_ != old_numel;
    }
    
    bool SetDims(OtterArrayRef<int64_t> d) {
        return SetDimsTemplate(d);
    }
    
    bool SetDims(OtterArrayRef<int> d) {
        return SetDimsTemplate(d);
    }
    
    bool SetDims(const int64_t d0) {
        return SetDimsTemplate(IntArrayRef{d0});
    }
    
    bool SetDims(const int64_t d0, const int64_t d1) {
        return SetDimsTemplate(IntArrayRef{d0, d1});
    }
    
    bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2) {
        return SetDimsTemplate(IntArrayRef{d0, d1, d2});
    }
    
    bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3) {
        return SetDimsTemplate(IntArrayRef{d0, d1, d2, d3});
    }
    
    template <typename... Ts>
    void Resize(Ts... dim_source) {
        bool size_changed = SetDims(dim_source...);
        if (size_changed) {
            this->HandleResize();
        }
    }
    
    template <typename T>
    void Resize(const std::vector<T>& dim_source) {
        Resize(OtterArrayRef<T>(dim_source));
    }
    
    void HandleResize() {
        bool reset_tensor = false;
        
        reset_tensor = memory_.nbytes() < (memory_offset_ + numel_) * data_type_.itemsize();
        
        if (reset_tensor && memory_initialized()) {
            this->FreeMemory();
        }
    }
    
    // retain for autograd
    
    //
private:
    Device device_;
    
    OtterMemory memory_;
    int64_t memory_offset_ = 0;
    int64_t numel_ = 1;
    
    OtterType data_type_;
    PerspectiveView perspective_view_;
};

class TensorBase {
public:
    struct unsafe_borrow_t { explicit unsafe_borrow_t() = default; };
protected:
    explicit TensorBase(unsafe_borrow_t, const TensorBase& rhs) : tensor_nucleus_(OtterPtr<TensorNucleus>::reclaim(rhs.tensor_nucleus_.get())) {}
    friend OtterMaybeOwnedTraits<TensorBase>;
public:
    TensorBase() = default;
    TensorBase(const TensorBase&) = default;
    TensorBase(TensorBase&&) = default;
    
    TensorBase& operator=(const TensorBase& t) & {
        tensor_nucleus_ = t.tensor_nucleus_;
        return *this;
    }
    
    TensorBase& operator=(TensorBase&& t) & {
        tensor_nucleus_ = std::move(t.tensor_nucleus_);
        return *this;
    }
    
    explicit TensorBase(OtterPtr<TensorNucleus> tensor_nucleus) : tensor_nucleus_(std::move(tensor_nucleus)) {
        if (tensor_nucleus_.get() == nullptr) {
            fprintf(stderr, "[TensorBase] Initialization failed!\n");
        }
    }
    
    int64_t dim() const {
        return tensor_nucleus_->dim();
    }
    
    TensorNucleus* unsafeGetTensorNucleus() const {
        return tensor_nucleus_.get();
    }
    
    TensorNucleus* unsafeReleaseTensorNucleus() {
        return tensor_nucleus_.release();
    }
    
    OtterPtr<TensorNucleus> getOtterPtr() const {
        return tensor_nucleus_;
    }
    
    OtterPtr<TensorNucleus> unsafeReleaseOtterPtr() {
        return std::move(tensor_nucleus_);
    }
    
    int64_t memory_offset() {
        return tensor_nucleus_->memory_offset();
    }
    
    bool defined() const {
        return tensor_nucleus_;
    }
    
    void reset() {
        tensor_nucleus_.reset();
    }
    
    bool is_same(const TensorBase& other) const {
        return tensor_nucleus_ == other.tensor_nucleus_;
    }
    
    size_t use_count() const noexcept {
        return tensor_nucleus_.use_count();
    }
    
    bool has_memory() const {
        return tensor_nucleus_->has_memory();
    }
    
    const OtterMemory& memory() const {
        return tensor_nucleus_->memory();
    }
    
    OtterType dtype() const noexcept {
        return tensor_nucleus_->dtype();
    }
    
    IntArrayRef sizes() const {
        return tensor_nucleus_->sizes();
    }
    
    IntArrayRef strides() const {
        return tensor_nucleus_->strides();
    }
    
    size_t nbytes() const {
        return tensor_nucleus_->numel() * tensor_nucleus_->itemsize();
    }
    
    int64_t numel() const {
        return tensor_nucleus_->numel();
    }
    
    size_t itemsize() const {
        return tensor_nucleus_->itemsize();
    }
    
    void print() const;
    
    std::string toString() const;
    
protected:
    OtterPtr<TensorNucleus> tensor_nucleus_;
};

template <typename T, typename... Args>
TensorBase make_tensor_base(Args&&... args) {
    return TensorBase(make_otterptr<T>(std::forward<Args>(args)...));
}

template <>
struct OtterMaybeOwnedTraits<TensorBase> {
    using owned_type = TensorBase;
    using borrow_type = TensorBase;
    
    static borrow_type create_borrow(const owned_type& from) {
        return borrow_type(borrow_type::unsafe_borrow_t{}, from);
    }
    
    static void assign_borrow(borrow_type& lhs, const borrow_type& rhs) {
        lhs.unsafeReleaseTensorNucleus();
        lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
    }

    static void destroy_borrow(borrow_type& target) {
        target.unsafeReleaseTensorNucleus();
    }

    static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
        return borrow;
    }

    static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
        return &borrow;
    }
};

class OtterTensorRef;

class Tensor : public TensorBase {
protected:
    explicit Tensor(unsafe_borrow_t, const TensorBase& rhs): TensorBase(unsafe_borrow_t{}, rhs) {}
    friend OtterMaybeOwnedTraits<Tensor>;
    friend OtterTensorRef;
public:
    Tensor() = default;

    explicit Tensor(OtterPtr<TensorNucleus> tensor_nucleus) : TensorBase(std::move(tensor_nucleus)) {}
    
    Tensor(const Tensor &tensor) = default;
    Tensor(Tensor &&tensor) = default;
    
    explicit Tensor(const TensorBase &base): TensorBase(base) {}
    Tensor(TensorBase &&base): TensorBase(std::move(base)) {}
    
    Tensor& operator=(const TensorBase& x) & {
        tensor_nucleus_ = x.getOtterPtr();
        return *this;
    }
    Tensor& operator=(TensorBase&& x) & {
        tensor_nucleus_ = x.unsafeReleaseOtterPtr();
        return *this;
    }

    Tensor& operator=(const Tensor &x) & {
        return operator=(static_cast<const TensorBase&>(x));
    }
    Tensor& operator=(Tensor &&x) & {
        return operator=(static_cast<TensorBase&&>(x));
    }
};

template <typename T, typename... Args>
Tensor make_tensor(Args&&... args) {
    return Tensor(make_otterptr<T>(std::forward<Args>(args)...));
}

template <>
struct OtterMaybeOwnedTraits<Tensor> {
    using owned_type = Tensor;
    using borrow_type = Tensor;

    static borrow_type create_borrow(const owned_type& from) {
        return borrow_type(borrow_type::unsafe_borrow_t{}, from);
    }

    static void assign_borrow(borrow_type& lhs, const borrow_type& rhs) {
        lhs.unsafeReleaseTensorNucleus();
        lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
    }

    static void destroy_borrow(borrow_type& toDestroy) {
        toDestroy.unsafeReleaseTensorNucleus();
    }

    static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
        return borrow;
    }

    static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
        return &borrow;
    }
};

}   // namespace otter

#endif /* OtterTensor_hpp */
