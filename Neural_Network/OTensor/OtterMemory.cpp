//
//  OtterMemory.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/11/19.
//

#include "OtterMemory.hpp"

OtterMemory_Tiny::~OtterMemory_Tiny() {
    if (cpu_data_ != nullptr && cpu_data_->unreference() == 0) {
        delete cpu_data_;
    }
}

OtterMemory_Tiny::OtterMemory_Tiny() : cpu_data_(nullptr), size_(0), state_(UNINITIALIZED) {}

OtterMemory_Tiny::OtterMemory_Tiny(size_t size) : cpu_data_(nullptr), size_(size), state_(UNINITIALIZED) {}

OtterMemory_Tiny::OtterMemory_Tiny(const OtterMemory_Tiny &other) {
    size_ = other.size_;
    cpu_data_ = other.cpu_data_;
    state_ =  (cpu_data_ && cpu_data_->reference()) ? REFERENCE_CPU : UNINITIALIZED;
}

OtterMemory_Tiny& OtterMemory_Tiny::operator=(const OtterMemory_Tiny& other) {
    if (this == &other) {
        return *this;
    }
    if (cpu_data_ != nullptr && cpu_data_->unreference() == 0) {
        delete cpu_data_;
    }
    size_ = other.size_;
    cpu_data_ = other.cpu_data_;
    state_ =  (cpu_data_ && cpu_data_->reference()) ? REFERENCE_CPU : UNINITIALIZED;
    
    return *this;
}

OtterMemory_Tiny OtterMemory_Tiny::clone() {
    OtterMemory_Tiny clone(this->size());
    if (state_ == UNINITIALIZED) return clone;
    
    this->copyTo(clone);
    
    return clone;
}

void OtterMemory_Tiny::copyFrom(OtterMemory_Tiny &other) {
    const void *src_data = other.cpu_data();
    void *dst_data = this->mutable_cpu_data();
    memcpy(dst_data, src_data, std::min(size(), other.size()));
}

void OtterMemory_Tiny::copyTo(OtterMemory_Tiny &other) {
    const void *src_data = this->cpu_data();
    void *dst_data = other.mutable_cpu_data();
    memcpy(dst_data, src_data, std::min(size(), other.size()));
}

void OtterMemory_Tiny::to_cpu() {
    switch (state_) {
        case UNINITIALIZED:
            cpu_data_ = new OtterData(size_);
            state_ = OWN_CPU;
            break;
        case OWN_CPU:
        case REFERENCE_CPU:
            break;
    }
}

const void* OtterMemory_Tiny::cpu_data() {
    to_cpu();
    return (const void*)cpu_data_->cpu_data();
}

void* OtterMemory_Tiny::mutable_cpu_data() {
    to_cpu();
    return (void*)cpu_data_->cpu_data();
}

void OtterMemory_Tiny::status() {
    std::cout << "<OtterMemory_Tiny at " << this << ">" << std::endl;
    std::cout << "-> Size: " << size() << "(bytes) Status: ";
    switch (state_) {
        case UNINITIALIZED: std::cout << "UNINITIALIZED"; break;
        case OWN_CPU: std::cout << "OWN_CPU"; break;
        case REFERENCE_CPU: std::cout << "REFERENCE_CPU"; break;
    }
    std::cout << " Physical: " << cpu_data_ << std::endl;
}
