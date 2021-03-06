//
//  OtterTensorOption.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/9.
//

#ifndef OtterTensorOption_hpp
#define OtterTensorOption_hpp

#include <stdio.h>

#include "OtterAllocator.hpp"
#include "OtterDType.hpp"

namespace otter {

struct TensorOption {
    TensorOption() : has_device_(false), has_data_type_(false), has_required_grad_(false) {}
    
    TensorOption(ScalarType dtype) : TensorOption() {
        this->set_dtype(dtype);
    }
    
    Device device() const noexcept {
        return device_;
    }
    
    bool has_device() const noexcept {
        return has_device_;
    }
    
    OtterType dtype() const noexcept {
        return data_type_;
    }
    
    bool has_dtype() const noexcept {
        return has_data_type_;
    }
    
    bool required_grad() const noexcept {
        return has_required_grad_ ? required_grad_ : false;
    }
    
    TensorOption device(Device device) const noexcept {
        TensorOption t = *this;
        t.set_device(device);
        return t;
    }
    
    TensorOption dtype(OtterType data_type) const noexcept {
        TensorOption t = *this;
        t.set_dtype(data_type);
        return t;
    }
    
    template <typename T>
    TensorOption& dtype() {
        data_type_ = OtterType::Make<T>();
        has_data_type_ = true;
        return *this;
    }
    
    TensorOption required_grad(bool required) const noexcept {
        TensorOption t = *this;
        t.set_required_grad(required);
        return t;
    }
    
private:
    void set_device(Device device) & noexcept {
        device_ = device;
        has_device_ = true;
    }
    
    void set_dtype(OtterType data_type) & noexcept {
        data_type_ = data_type;
        has_data_type_ = true;
    }
    
    void set_dtype(ScalarType dtype) & noexcept {
        data_type_ = scalarTypeToTypeMeta(dtype);
        has_data_type_ = true;
    }
    
    void set_required_grad(bool required_grad) & noexcept {
        required_grad_ = required_grad;
        has_required_grad_ = true;
    }
    
    
    Device device_ = Device::CPU;
    OtterType data_type_ = OtterType::Make<float>();
    
    bool required_grad_ : 1;
    
    bool has_device_ : 1;
    bool has_data_type_ : 1;
    bool has_required_grad_ : 1;
};

}   // end namespace otter


#endif /* OtterTensorOption_hpp */
