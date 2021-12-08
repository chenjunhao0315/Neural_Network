//
//  OtterMemory.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/11/19.
//

#ifndef OtterMemory_hpp
#define OtterMemory_hpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>

#include "OtterAllocator.hpp"

class OtterRef;
class OtterData;
class OtterMemory_Tiny;

class OtterMemory_Tiny {
public:
    ~OtterMemory_Tiny();
    OtterMemory_Tiny();
    OtterMemory_Tiny(size_t size);
    OtterMemory_Tiny(const OtterMemory_Tiny& other);
    OtterMemory_Tiny& operator=(const OtterMemory_Tiny& other);
    OtterMemory_Tiny clone();
    
    void copyFrom(OtterMemory_Tiny &other);
    void copyTo(OtterMemory_Tiny &other);
    
    const void *cpu_data();
    void *mutable_cpu_data();
    
    enum MemoryState { UNINITIALIZED, OWN_CPU, REFERENCE_CPU };
    MemoryState state() { return state_; }
    size_t size() const { return size_; }
    void status();
private:
    void to_cpu();
    
    OtterData *cpu_data_;
    MemoryState state_;
    size_t size_;
};

class OtterRef {
public:
    OtterRef() : refCount_(1) {}
    int reference() { return ++refCount_; }
    int unreference() { return --refCount_; }
    int refCount() { return refCount_; }
private:
    int refCount_;
};

class OtterData : public OtterRef {
public:
    ~OtterData() { otter_free(cpu_data_); }
    OtterData() : cpu_data_(nullptr), OtterRef() {}
    OtterData(size_t size) : OtterData() { cpu_data_ = otter_calloc(1, size); }
    void *cpu_data() { return cpu_data_; }
private:
    void *cpu_data_;
};

class OtterPtr_quantum {
    template <class T>
    friend class OtterPtr;
    
protected:
    virtual ~OtterPtr_quantum() {}
    constexpr OtterPtr_quantum() noexcept : refCount_(0) {}
    OtterPtr_quantum(const OtterPtr_quantum&& other) noexcept : OtterPtr_quantum() {}
    OtterPtr_quantum& operator=(const OtterPtr_quantum&& other) { return *this; }
    OtterPtr_quantum(const OtterPtr_quantum& other) noexcept : OtterPtr_quantum() {}
    OtterPtr_quantum& operator=(const OtterPtr_quantum& other) { return *this; }
    
private:
    virtual void release_resources() {}
    
    int refCount_;
};

struct DontIncreaseRefcount {};

template <class Target>
class OtterPtr {
public:
    OtterPtr() noexcept : OtterPtr(nullptr, DontIncreaseRefcount{}) {}
    
    explicit OtterPtr(Target *target, DontIncreaseRefcount) noexcept : target_(target) {}
    
    OtterPtr(const OtterPtr& other) : target_(other.target_) {
        retain_();
    }
    OtterPtr(OtterPtr&& other) : target_(other.target_) {
        other.target_ = nullptr;
    }
    ~OtterPtr() noexcept {
        reset();
    }
    
    OtterPtr& operator=(OtterPtr&& rhs) & noexcept {
        return operator=<Target>(std::move(rhs));
    }

    template <class From>
    OtterPtr& operator=(OtterPtr<From>&& rhs) & noexcept {
        OtterPtr tmp = std::move(rhs);
        this->swap(tmp);
        return *this;
    }

    OtterPtr& operator=(const OtterPtr& rhs) & noexcept {
        return operator=<Target>(rhs);
    }

    template <class From>
    OtterPtr& operator=(const OtterPtr<From>& rhs) & {
        OtterPtr tmp = rhs;
        this->swap(tmp);
        return *this;
    }
    
    const Target& operator*() const noexcept { return *target_; }
    Target& operator*() noexcept { return *target_; }
    const Target* operator->() const noexcept { return target_; }
    Target* operator->() noexcept { return target_; }
    operator bool() const noexcept { return target_ != nullptr; }
    
    Target* get() const noexcept {
        return target_;
    }
    
    void reset() {
        reset_();
    }
    
    int use_count() const noexcept {
        return (target_) ? target_->refCount_ : 0;
    }
    
    bool unique() {
        return use_count() == 1;
    }
    
    bool defined() const noexcept {
        return target_ != nullptr;
    }
    
    void swap(OtterPtr &other) {
        Target* temp = target_;
        target_ = other.target_;
        other.target_ = temp;
    }
    
    // The refCount is not decreased!
    Target* release() noexcept {
        Target* result = target_;
        target_ = nullptr;
        return result;
    }
    
    // The refCount is not increased!
    static OtterPtr reclaim(Target* ptr) {
        return OtterPtr(ptr, DontIncreaseRefcount{});
    }
    
    template <class... Args>
    static OtterPtr make(Args&&... args) {
        return OtterPtr(new Target(std::forward<Args>(args)...));
    }
    
private:
    void retain_() {
        if (target_ != nullptr)
            ++target_->refCount_;
    }
    void reset_() {
        if (target_ != nullptr && --target_->refCount_ == 0) {
            target_->release_resources();
            delete target_;
        }
        target_ = nullptr;
    }
    
    explicit OtterPtr(Target *target) : target_(target) {
        if (target != nullptr) {
            target_->refCount_ = 1;
        }
    }
    
    Target *target_;
};

template <class Target, class... Args>
inline OtterPtr<Target> make_otterptr(Args&&... args) {
    return OtterPtr<Target>::make(std::forward<Args>(args)...);
}

class MemoryNucleus : public OtterPtr_quantum {
public:
    MemoryNucleus(size_t size_bytes, DataPtr data_ptr, Allocator* allocator) : size_bytes_(size_bytes), data_ptr_(std::move(data_ptr)), allocator_(allocator) {}
    MemoryNucleus(size_t size_bytes, Allocator* allocator) : MemoryNucleus(size_bytes, allocator->allocate(size_bytes), allocator) {}
    
    MemoryNucleus() = delete;
    MemoryNucleus(const MemoryNucleus&) = delete;
    MemoryNucleus(MemoryNucleus&& other) = default;
    ~MemoryNucleus() override = default;
    MemoryNucleus& operator=(MemoryNucleus&& other) = default;
    MemoryNucleus& operator=(const MemoryNucleus&) = delete;
    
    void reset() {
        data_ptr_.clear();
        size_bytes_ = 0;
    }
    
    template <typename T>
    inline T* data() const {
        return unsafe_data<T>();
    }

    template <typename T>
    inline T* unsafe_data() const {
        return static_cast<T*>(this->data_ptr_.get());
    }

    void release_resources() override {
        data_ptr_.clear();
    }

    size_t nbytes() const {
        return size_bytes_;
    }
    
    void set_nbytes(size_t size_bytes) {
        size_bytes_ = size_bytes;
    }
    
    DataPtr& data_ptr() {
        return data_ptr_;
    }
    
    const DataPtr& data_ptr() const {
        return data_ptr_;
    }
    
    DataPtr set_data_ptr(DataPtr&& data_ptr) {
        std::swap(data_ptr_, data_ptr);
        return std::move(data_ptr);
    };

    void set_data_ptr_noswap(DataPtr&& data_ptr) {
        data_ptr_ = std::move(data_ptr);
    }
    
    void* data() {
        return data_ptr_.get();
    }
    
    void* data() const {
        return data_ptr_.get();
    }
    
    Allocator* allocator() {
        return allocator_;
    }
    
    const Allocator* allocator() const {
        return allocator_;
    }
    
    Device device() const {
        return data_ptr_.device();
    }
    
private:
    DataPtr data_ptr_;
    size_t size_bytes_;
    Allocator* allocator_;
};

struct OtterMemory {
public:
    OtterMemory() {}
    OtterMemory(OtterPtr<MemoryNucleus> ptr) : memory_nucleus_(std::move(ptr)) {}
    OtterMemory(size_t size, Allocator* allocator) : memory_nucleus_(make_otterptr<MemoryNucleus>(size, allocator)) {}
    
    operator bool() const {
        return memory_nucleus_;
    }
    
    template <typename T>
    inline T* data() const {
        return memory_nucleus_->data<T>();
    }

    template <typename T>
    inline T* unsafe_data() const {
        return memory_nucleus_->unsafe_data<T>();
    }

    size_t nbytes() const {
        return memory_nucleus_->nbytes();
    }
    
    void set_nbytes(size_t size_bytes) const {
        memory_nucleus_.get()->set_nbytes(size_bytes);
    }
    
    void* data() const {
        return memory_nucleus_.get()->data();
    }
    
    DataPtr& data_ptr() {
        return memory_nucleus_->data_ptr();
    }
    
    const DataPtr& data_ptr() const {
        return memory_nucleus_->data_ptr();
    }
    
    DataPtr set_data_ptr(DataPtr&& data_ptr) const {
        return memory_nucleus_.get()->set_data_ptr(std::move(data_ptr));
    }
    
    void set_data_ptr_noswap(DataPtr&& data_ptr) const {
        return memory_nucleus_.get()->set_data_ptr_noswap(std::move(data_ptr));
    }
    
    Allocator* allocator() const {
        return memory_nucleus_.get()->allocator();
    }
    
    Device device() const {
        return memory_nucleus_->device();
    }
    
    static OtterMemory create_empty(Device device) {
        Allocator* allocator = GetAllocator(device);
        return OtterMemory(make_otterptr<MemoryNucleus>(0, allocator->allocate(0), allocator));
    }
    
protected:
    OtterPtr<MemoryNucleus> memory_nucleus_;
};

#endif /* OtterMemory_hpp */
